import flwr as fl
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import tensorflow as tf
import os
import random
import tenseal as ts
import pickle
from dataset_utils import ManageDatasets, load_data_flowerdataset

import tracemalloc
import sys
import pickle
import time

from models import create_alexnet, create_dnn, create_lenet5, reshape_parameters, flat_parameters
from client_logs import write_train_logs, write_evaluate_logs
# import logging
# logging.basicConfig(level=logging.DEBUG)

class HEClient(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, 
                 dirichlet_alpha, start2share, homomorphic, homomorphic_type):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.last_parameters  = None
        self.dataset_manager  = ManageDatasets(self.cid)
        self.start2share      = start2share
        self.homomorphic      = homomorphic
        self.homomorphic_type = homomorphic_type                                    

        if dataset == 'MNIST' or dataset == 'CIFAR10':
            self.x_train, self.y_train, self.x_test, self.y_test = load_data_flowerdataset(self)
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_har(dataset) #self.load_data()
            
        if dataset == 'CIFAR10':
            self.model  = create_lenet5(self.x_train.shape, len(np.unique(self.y_train)))

        else:
            self.model  = create_dnn(self.x_train.shape, len(np.unique(self.y_train)))
        
        if self.homomorphic:
            self.context = self.get_client_context()
        
        
    def get_client_context(self):
        if self.homomorphic_type == 'Full':
            with open(f'../context/secret.pkl', 'rb') as file:
                secret = pickle.load(file)
        else:
            with open(f'../context/secret_partial.pkl', 'rb') as file:
                secret = pickle.load(file)  

        context = ts.context_from(secret["context"])

        return context

    def get_parameters(self, config):
        parameters           = self.model.get_weights()
        return parameters
    
    def load_har(self, dataset):
        if dataset == 'UCIHAR':
            return self.dataset_manager.load_UCIHAR()
        if dataset == 'ExtraSensory':
            return self.dataset_manager.load_ExtraSensory()
        if dataset == 'MotionSense':
            return self.dataset_manager.load_MotionSense()

    
    def he_parameters_to_model(self, he_parameters):
        local_parameters     = self.model.get_weights()
        decrypted_parameters = he_parameters.decrypt()
        reshaped_parameters  = reshape_parameters(decrypted_parameters)
        self.model.set_weights(reshaped_parameters)
        # temp_flat            = self.flat_parameters(local_parameters[:self.start2share])
        # temp_flat.extend(decrypted_parameters)

    def fit(self, parameters, config):
        
        decyfer_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            he_parameters        = ts.ckks_vector_from(self.context, config['he'])
            self.he_parameters_to_model(he_parameters)
            
        else:
            self.model.set_weights(parameters)
        decyfer_time = time.time() - decyfer_time

        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()
        he_parameters      = []
        cyfer_time         = time.time()
        
        if self.homomorphic:
            flatted_parameters = flat_parameters(trained_parameters) 
            he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
            he_parameters      = he_parameters.serialize()
            model_size         = sys.getsizeof(he_parameters) 
        
        else:
            flatted_parameters  = flat_parameters(trained_parameters)
            temp_buf            = pickle.dumps(flatted_parameters)
            model_size          = sys.getsizeof(temp_buf)
        
        cyfer_time = time.time() - cyfer_time
        
        write_train_logs(self, config['round'], loss, acc, model_size, train_time, cyfer_time, decyfer_time)
       
        fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters)
       
        return trained_parameters, len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decyfer_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            client_context = self.get_client_context()
            he_parameters        = ts.ckks_vector_from(client_context, config['he'])
            self.he_parameters_to_model(he_parameters)
            
        else:
            self.model.set_weights(parameters)
        decyfer_time = time.time() - decyfer_time
        loss, acc = self.model.evaluate(self.x_test, self.y_test)

        write_evaluate_logs(self, config['round'], loss, acc, decyfer_time)

        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }

        return loss, len(self.x_test), eval_msg
    
    def create_fit_msg(self, train_time, acc, loss, model_size, he_parameters):
        
        fit_msg = {
            'cid'         : self.cid,
            'train_time'  : train_time,
            'accuracy'    : acc,
            'loss'        : loss,
            'delay_start' : time.time(),
            'data_size'   : model_size,
            'he'          : he_parameters if self.homomorphic else ''
        }
        
        return fit_msg
        
    

def main():
    
    client =  HEClient(
                        cid             = int(os.environ['CID']), 
                        niid            = os.environ['NIID'] == 'True', 
                        dataset         = os.environ['DATASET'], 
                        num_clients     = int(os.environ['NCLIENTS']), 
                        dirichlet_alpha = float(os.environ['DIRICHLET_ALPHA']),
                        start2share     = int(os.environ['START2SHARE']),
                        homomorphic     = os.environ['HOMOMORPHIC'] == 'True',
                        homomorphic_type= str(os.environ['HOMOMORPHIC_TYPE'])
                        )
        
    fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                                client=client)


if __name__ == '__main__':
	main()