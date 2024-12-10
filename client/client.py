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

from encryption.quantize import quantize, unquantize, batch_padding, unbatching_padding
from encryption.paillier import PaillierCipher

from literature import fit_ckks, fit_batchcrypt, fit_fedphe, fit_plaintext, he_packs_to_model, he_parameters_to_model
# import logging
# logging.basicConfig(level=logging.DEBUG)

class HEClient(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, 
                 dirichlet_alpha, start2share, solution):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.last_parameters  = None
        self.dataset_manager  = ManageDatasets(self.cid)
        self.start2share      = start2share
        self.solution         = str(solution).lower()
        # self.homomorphic      = homomorphic
        # self.packing          = packing
        # self.only_sum         = onlysum
        # self.homomorphic_type = homomorphic_type   
        
        self.len_shared_data  = 0                                 

        if dataset == 'MNIST' or dataset == 'CIFAR10':
            self.x_train, self.y_train, self.x_test, self.y_test = load_data_flowerdataset(self)
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_har(dataset) #self.load_data()
            
        if dataset == 'CIFAR10':
            self.model  = create_lenet5(self.x_train.shape, len(np.unique(self.y_train)))

        else:
            self.model  = create_dnn(self.x_train.shape, len(np.unique(self.y_train)))
        
        self.config_solution()
        if self.homomorphic:
            self.context = self.get_client_context()
        
        
    def get_client_context(self):
        if self.homomorphic_type == 'Paillier':
            with open(f'context/paillier.pkl', 'rb') as file:
                context = pickle.load(file)    
        else:
            with open(f'../context/secret.pkl', 'rb') as file:
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
 
    def config_solution(self):
        
        if str(self.solution).lower() == 'ckks':
            self.homomorphic      = True
            self.packing          = False
            self.only_sum         = False
            self.homomorphic_type = 'CKKS'
        
        elif str(self.solution).lower() == 'batchcrypt':
            self.homomorphic      = True
            self.homomorphic_type = 'Paillier'
            self.only_sum         = True
            self.packing          = False
            
        elif str(self.solution).lower() == 'fedphe':
            self.homomorphic      = True
            self.homomorphic_type = 'CKKS'
            self.only_sum         = False
            self.packing          = True
            
        elif str(self.solution).lower() == 'plaintext':
            self.homomorphic      = False
            self.only_sum         = False
            self.packing          = False
            self.homomorphic_type = 'None'       

    def fit(self, parameters, config):
        
        if str(self.solution).lower() == 'ckks':
            fit_msg = fit_ckks(self, parameters, config)
            
        elif str(self.solution).lower() == 'batchcrypt':
            fit_msg = fit_batchcrypt(self, parameters, config)
            
        elif str(self.solution).lower() == 'fedphe':
            fit_msg = fit_fedphe(self, parameters, config)
            
        elif str(self.solution).lower() == 'plaintext':
            fit_msg = fit_plaintext(self, parameters, config)
       
        return self.model.get_weights(), len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                he_packs_to_model(self, config)
                
            else:
                he_parameters_to_model(self, config)
            
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
                        solution        = str(os.environ['SOLUTION'])
                        # homomorphic     = os.environ['HOMOMORPHIC'] == 'True',
                        # packing         = os.environ['PACKING'] == 'True',
                        # onlysum         = os.environ['ONLYSUM'] == 'True',
                        # homomorphic_type= str(os.environ['HOMOMORPHIC_TYPE'])
                        )
        
    fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                                client=client)


if __name__ == '__main__':
	main()