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

from models import create_cnn, create_dnn, create_lenet5, reshape_parameters, flat_parameters,ResNet20,CNN_fmnist_tf
from client_logs import write_train_logs, write_evaluate_logs
from client_utils import get_size, packing, cypher_packs, get_topk_mask, decypher_packs, flat_packs, remove_padding

from encryption.quantize import quantize, unquantize, batch_padding, unbatching_padding
from encryption.paillier import PaillierCipher

from literature import fit_ckks, fit_bfv, fit_batchcrypt, fit_fedphe, fit_plaintext, fit_yphe, he_packs_to_model,he_packs_to_model_yphe, he_parameters_to_model
# import logging
# logging.basicConfig(level=logging.DEBUG)

# import logging

from pathlib import Path

LOG_DIR = "logs/"


def get_latest_created_folder(directory):
    # Lista todos os diretórios dentro do caminho especificado
    folders = [f for f in Path(directory).iterdir() if f.is_dir()]
    
    # Se houver diretórios, ordene-os pela data de criação (metadata de criação - st_ctime)
    if folders:
        latest_folder = max(folders, key=lambda f: f.stat().st_ctime)
        return latest_folder
    else:
        return None


class HEClient(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, 
                 dirichlet_alpha, start2share, solution,percentage,technique):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.last_parameters  = None
        self.dataset_manager  = ManageDatasets(self.cid)
        self.start2share      = start2share
        self.solution         = str(solution).lower()
        self.weights_packs    = []
        self.technique        = technique
        self.percentage       = percentage
        self.dirichlet_alpha = dirichlet_alpha
        # self.homomorphic      = homomorphic
        # self.packing          = packing
        # self.only_sum         = onlysum
        # self.homomorphic_type = homomorphic_type   

        if dataset == 'MNIST' or dataset == 'CIFAR10' or dataset == "FASHION_MNIST":
            self.x_train, self.y_train, self.x_test, self.y_test = load_data_flowerdataset(self)
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_har(dataset) #self.load_data()
        print("\n\n\n\n\nSHAPE:",self.x_train.shape)
        
        if dataset == 'FASHION_MNIST':
            self.model = create_lenet5(self.x_train.shape,10)
            self.model_name = "lenet5"
        elif dataset == "MNIST":
            self.model = create_dnn(self.x_train.shape,10)
            self.model_name = "DNN"
        else:
            return
        
        self.len_shared_data  =  len(flat_parameters(self.model.get_weights()))                               
        
        self.config_solution()
        if self.homomorphic:
            self.context = self.get_client_context()
        
        
    def get_client_context(self):
        if self.homomorphic_type == 'Paillier':
            with open(f'context/paillier.pkl', 'rb') as file:
                context = pickle.load(file)    
        else:
            if 'CKKS' == self.homomorphic_type:
                with open(f'../context/ckks_secret.pkl', 'rb') as file:
                    secret = pickle.load(file) 
                    context = ts.context_from(secret["context"])
            else:
                with open(f'../context/bfv_secret.pkl', 'rb') as file:
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
            
        elif str(self.solution).lower() == 'bfv':
            self.homomorphic      = True
            self.packing          = False
            self.only_sum         = True
            self.homomorphic_type = 'BFV'
        
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
        elif str(self.solution).lower() == 'yphe':
            self.homomorphic      = True 
            self.only_sum         = False
            self.homomorphic_type = 'CKKS'
            self.packing          = True
        elif str(self.solution).lower() == 'plaintext':
            self.homomorphic      = False
            self.only_sum         = False
            self.packing          = False
            self.homomorphic_type = 'None'

    def fit(self, parameters, config):
        
        if str(self.solution).lower() == 'ckks':
            fit_msg = fit_ckks(self, parameters, config)
            
        elif str(self.solution).lower() == 'bfv':
            fit_msg = fit_bfv(self, parameters, config)
            
        elif str(self.solution).lower() == 'batchcrypt':
            fit_msg = fit_batchcrypt(self, parameters, config)
            
        elif str(self.solution).lower() == 'fedphe':
            fit_msg = fit_fedphe(self, parameters, config)
        elif str(self.solution).lower() == 'yphe':
            fit_msg = fit_yphe(self,parameters,config)
            
        elif str(self.solution).lower() == 'plaintext':
            fit_msg = fit_plaintext(self, parameters, config)
       
        return self.model.get_weights(), len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                if str(self.solution).lower() == 'yphe':
                    he_packs_to_model_yphe(self,config)
                else: 
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
        # with open(f'{self.log_folder}/client_{self.cid}_eval.csv', 'a') as f:
        #     f.write(f"{acc},{loss}\n")


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
                        solution        = str(os.environ['SOLUTION']),        
                        percentage      = float(os.environ['PERCENTAGE']),
                        technique       = str(os.environ["TECHNIQUE"]),
                                        #   "robin_round"
                                        #   "slided_window"
                                        #   "weight_random"
                        # homomorphic     = os.environ['HOMOMORPHIC'] == 'True',
                        # packing         = os.environ['PACKING'] == 'True',
                        # onlysum         = os.environ['ONLYSUM'] == 'True',
                        # homomorphic_type= str(os.environ['HOMOMORPHIC_TYPE'])
                        )
        
    fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                                client=client)


if __name__ == '__main__':
	main()