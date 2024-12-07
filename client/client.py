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
import time

from models import create_cnn, create_dnn, create_lenet5, reshape_parameters, flat_parameters
from client_logs import write_train_logs, write_evaluate_logs
from client_utils import get_size, packing, cypher_packs, get_topk_mask, decypher_packs, flat_packs, remove_padding

# import logging
# logging.basicConfig(level=logging.DEBUG)

class HEClient(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, 
                 dirichlet_alpha, start2share, homomorphic, packing, homomorphic_type):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.last_parameters  = None
        self.dataset_manager  = ManageDatasets(self.cid)
        self.start2share      = start2share
        self.homomorphic      = homomorphic
        self.packing          = packing
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
 
    def he_parameters_to_model(self, config):
        he_parameters        = ts.ckks_vector_from(self.context, config['he'])
        local_parameters     = self.model.get_weights()
        decrypted_parameters = he_parameters.decrypt()
        reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
        self.model.set_weights(reshaped_parameters)
        # temp_flat            = self.flat_parameters(local_parameters[:self.start2share])
        # temp_flat.extend(decrypted_parameters)

    def fit(self, parameters, config):
        
        decypher_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                self.he_packs_to_model(config)
            else:
                self.he_parameters_to_model(config)
            
        else:
            self.model.set_weights(parameters)
        decypher_time = time.time() - decypher_time

        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()
        he_parameters      = []
        cypher_time         = time.time()
        
        if self.packing: 
            packed_parameters = packing(flat_parameters(trained_parameters))
            topk_mask         = get_topk_mask(packed_parameters, 0.1)
            cyphered_packs    = cypher_packs(packed_parameters, topk_mask, self.context)
            he_parameters     = pickle.dumps(cyphered_packs)
            model_size        = get_size(cyphered_packs)

        elif self.homomorphic and not self.packing:
            flatted_parameters = flat_parameters(trained_parameters) 
            he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
            he_parameters      = he_parameters.serialize()
            model_size         = sys.getsizeof(he_parameters)
            topk_mask          =  '' 
        
        else:
            flatted_parameters  = flat_parameters(trained_parameters)
            temp_buf            = pickle.dumps(flatted_parameters)
            model_size          = sys.getsizeof(temp_buf)
            topk_mask           =  '' 
        
        cypher_time = time.time() - cypher_time
        
        write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
       
        fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
       
        return trained_parameters, len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                self.he_packs_to_model(config)
                
            else:
                self.he_parameters_to_model(config)
            
        else:
            self.model.set_weights(parameters)
        decypher_time = time.time() - decypher_time
        loss, acc = self.model.evaluate(self.x_test, self.y_test)

        write_evaluate_logs(self, config['round'], loss, acc, decypher_time)

        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }

        return loss, len(self.x_test), eval_msg
    
    def he_packs_to_model(self, config):
        packed_parameters    = packing(flat_parameters(self.model.get_weights()))
        cyphered_packs        = pickle.loads(config['he'])
        decyphered_pack       = decypher_packs(cyphered_packs, self.context)
        aggredated_mask      = pickle.loads(config['mask'])
        
        for idx_mask, m in enumerate(aggredated_mask):
            if m > 0:
                packed_parameters[idx_mask] = decyphered_pack.pop(0)
                
        flatted_packs       = flat_packs(packed_parameters)
        flatted_packs       = remove_padding(self, flatted_packs)
        reshaped_parameters = reshape_parameters(self, flatted_packs)
        self.model.set_weights(reshaped_parameters)
    
    def create_fit_msg(self, train_time, acc, loss, model_size, he_parameters, mask=''):
        
        fit_msg = {
            'cid'         : self.cid,
            'train_time'  : train_time,
            'accuracy'    : acc,
            'loss'        : loss,
            'delay_start' : time.time(),
            'data_size'   : model_size,
            'he'          : he_parameters if self.homomorphic else '',
            'mask'        : pickle.dumps(mask)
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
                        packing         = os.environ['PACKING'] == 'True',
                        homomorphic_type= str(os.environ['HOMOMORPHIC_TYPE'])
                        )
        
    fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                                client=client)


if __name__ == '__main__':
	main()