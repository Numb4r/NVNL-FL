import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,models
import os
import random
import tenseal as ts
import pickle
import numpy as np
import time 
import sys
import math
from common.models.CNN import get_model
from common.common import top_k,load_data,flat_parameters,reshape_parameters,get_latest_created_folder
# logging.basicConfig(level=logging.DEBUG)

# import logging

from pathlib import Path
LOG_DIR = "logs/"




class FEDPHE_client(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha):
        
        
        self.log_folder = get_latest_created_folder(LOG_DIR)
        self.cid             = int(cid)
        self.dataset         = dataset
        self.niid            = niid
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.last_parameters = None                                   

        self.x_train, self.y_train, self.x_test, self.y_test = load_data(
                                                                            niid=self.niid,
                                                                            num_clients=self.num_clients,
                                                                            dataset=self.dataset,
                                                                            dirichlet_alpha=self.dirichlet_alpha,
                                                                            cid=self.cid
                                                                            )
        self.model                                           = self.create_model(self.x_train.shape)
        self.context                                         = self.get_client_context()
        self.k = 5
        self.mask = []
        self.QNT_BLOCOS = 14
        
        
    def get_client_context(self):
        with open(f'context/secret.pkl', 'rb') as file:
            secret = pickle.load(file)  

        context = ts.context_from(secret["context"])

        return context

    def get_parameters(self, config):
        parameters           = self.model.get_weights()
        return parameters
    
    def create_model(self, input_shape):
        model = get_model()
        return model

    def set_parameters(self,parameters,config):
        
        if len(config['he']) > 0:
            neurons = []
            for idx,layer in enumerate(self.model.layers):    
                weight = layer.get_weights()
                if weight:
                    flat = flat_parameters(weight[0])
                    neurons.extend(flat) # pesos
                    neurons.extend(weight[1]) # bias
            he = config['he']
            size_arrays_bytes= pickle.loads(config['size_array_bytes'])
            local_parameters     = self.model.get_weights()
            he_parameters= [] 
            idx = 0
            local_parameters=self.model.get_weights()
            TAMANHO_BLOCO = int(len(neurons)/self.QNT_BLOCOS)
            

            for size in size_arrays_bytes:
                he_parameters.append(he[idx:idx+size])
                idx = idx+size



            for idx in range(self.QNT_BLOCOS):
                if self.mask[idx] == 1:
                    decrypt_layer = ts.ckks_vector_from(self.context,he_parameters[idx]).decrypt()
                    neurons[idx*TAMANHO_BLOCO:idx*TAMANHO_BLOCO+TAMANHO_BLOCO] = decrypt_layer
                    # reshaped = self.reshape_parameters(decrypt_layer,local_parameters[idx])
                    # local_parameters[idx] = reshaped
            reshaped = reshape_parameters(neurons,local_parameters)
            local_parameters = reshaped
                # local_parameters[idx] = np.array(local_parameters[idx])
            self.model.set_weights(local_parameters)
            # for idx,layer in he_parameters:
            #     if layer != [0]:
            #         decrypt_layer = ts.ckks_vector_from(self.context,layer)
            #         reshaped = self.reshape_parameters(decrypt_layer,local_parameters[idx])
            #         local_parameters[idx] = reshaped

            # for idx_decrypt,idx_local in enumerate(mask):
            #     reshaped = self.reshape_parameters(decrypted_parameters[idx_decrypt],local_parameters[idx_local])

            #     local_parameters[idx_local] = reshaped

        
            
            # self.model.set_weights(local_parameters)
        
        
    def fit(self, parameters, config):
        
        print(f"Client {self.cid} - {config.keys()} - {[len(x) for x in config.items()]}")
        tempo_cifragem_total = 0
        tempo_decrypt_total = 0 
        self.set_parameters(parameters,config)
        
        time_training = time.time()


        history = self.model.fit(self.x_train, self.y_train, epochs=1)
        time_training_total = time.time() - time_training
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        
        neurons = []
        


        for idx,layer in enumerate(self.model.layers):    
            weight = layer.get_weights()
            if weight:
                flat = flat_parameters(weight[0])
                neurons.extend(flat) # pesos
                neurons.extend(weight[1]) # bias

        # print(f"TAMANHO MODEL LAYERS{len(neurons)}")
        # assert len(neurons) / 8 == math.floor(len(neurons) / 14)
        
        TAMANHO_BLOCO = int(len(neurons)/self.QNT_BLOCOS)
        print(f"TAMANHO DO BLOCO {TAMANHO_BLOCO}")
        idx = 0
        blocks =[]

        for i in range(self.QNT_BLOCOS):
            limit = idx + TAMANHO_BLOCO
            blocks.append(neurons[idx:limit])
            idx=limit
        
        
            
        selected_layers,self.mask = top_k(blocks,self.k,metric="norm")
        selected_layers = sorted(selected_layers)
        
        print(f"Selected Layers {selected_layers}")
        
        # trained_parameters = self.model.get_weights()
        
        send_layers = []
        for l in selected_layers:
            send_layers.append(blocks[l])
        
        tamanho_flat_parameter = sum([sys.getsizeof(s)  for s in send_layers])
        
        size_array_bytes= []
        tempo_cifragem_inicio = time.time()


        for idx,l in enumerate(send_layers):
            send_layers[idx] = ts.ckks_vector(self.context,l).serialize()
            size_array_bytes.append(len(send_layers[idx]))
        print(size_array_bytes)
        tempo_cifragem_total = time.time() - tempo_cifragem_inicio
        

        
        tamanho_cifrado =      sum([sys.getsizeof(s) for s in send_layers])

        
        
        
        

        he = b''
        for b in send_layers:
            he+=b

        fit_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss,
            'he'      : he,
            'mask'     : pickle.dumps(self.mask),
            'size_array_bytes':pickle.dumps(size_array_bytes),
        }
        print(f"Selected layers {selected_layers}")
        print(f"Mask {self.mask}")
        print(f"acc : {acc}")
        print(f"loss : {loss}")
        print(f"tamanho he: {len(he)}")
        print(f"tamanho_flat_parameter : {tamanho_flat_parameter}")
        print(f"tamanho_cifrado : {tamanho_cifrado}")
        print(f"tempo_cifragem_total : {tempo_cifragem_total}")
        print(f"tempo_decrypt_total : {tempo_decrypt_total}")
        print(f"time_training_tota : {time_training_total}")
        
        with open(f'{self.log_folder}/client_{self.cid}_train.csv', 'a') as f:
            f.write(f"{acc},{loss},{tamanho_flat_parameter},{tamanho_cifrado},{tempo_cifragem_total},{tempo_decrypt_total},{time_training_total} \n")
        return [], len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        
        print("VALIDAÇÃO")
        if len(config['he']) > 0:
            neurons = []
            for idx,layer in enumerate(self.model.layers):    
                weight = layer.get_weights()
                if weight:
                    flat = flat_parameters(weight[0])
                    neurons.extend(flat) # pesos
                    neurons.extend(weight[1]) # bias
            he = config['he']
            size_arrays_bytes= pickle.loads(config['size_array_bytes'])
            local_parameters     = self.model.get_weights()
            he_parameters= [] 
            idx = 0
            local_parameters=self.model.get_weights()
            TAMANHO_BLOCO = int(len(neurons)/self.QNT_BLOCOS)

            for size in size_arrays_bytes:
                he_parameters.append(he[idx:idx+size])
                idx = idx+size



            for idx in range(self.QNT_BLOCOS):
                if self.mask[idx] == 1:
                    decrypt_layer = ts.ckks_vector_from(self.context,he_parameters[idx]).decrypt()
                    neurons[idx*TAMANHO_BLOCO:idx*TAMANHO_BLOCO+TAMANHO_BLOCO] = decrypt_layer
                    # reshaped = self.reshape_parameters(decrypt_layer,local_parameters[idx])
                    # local_parameters[idx] = reshaped
            reshaped = reshape_parameters(neurons,local_parameters)
            local_parameters = reshaped
                # local_parameters[idx] = np.array(local_parameters[idx])
            self.model.set_weights(local_parameters)


        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        with open(f'{self.log_folder}/client_{self.cid}_eval.csv', 'a') as f:
            f.write(f"{acc},{loss}\n")
        

        return loss, len(self.x_test), eval_msg
    