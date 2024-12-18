import tenseal as ts
import numpy as np
import sys 
import time

from base_client import BaseNumpyClient
from client.encryption.quantize import unquantize,quantize
from client.common.utils import reshape_parameters,flat_parameters,create_fit_msg
from client.common.client_logs import write_evaluate_logs, write_train_logs

class BFVClient(BaseNumpyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager,only_sum=False):
        super().__init__(cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager)
        self.solution = 'bfv'
        self.only_sum = only_sum

    def fit(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 :
            self.set_parameters(config)
        decypher_time = time.time() - decypher_time

        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc        = np.mean(history.history['accuracy'])
        loss       = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()
        he_parameters      = []
        cypher_time        = time.time()
        

        if self.homomorphic:
            flatted_parameters = flat_parameters(trained_parameters) 
            
            if self.only_sum:
                flatted_parameters = np.array(flatted_parameters) #* len(self.x_train)
                
            quantized_parameters = quantize(flatted_parameters, 16, self.num_clients)
            he_parameters        = ts.bfv_vector(self.context, quantized_parameters)
            he_parameters        = he_parameters.serialize()
            model_size           = sys.getsizeof(he_parameters)
            topk_mask            =  '' 
        
        cypher_time = time.time() - cypher_time
        
        write_train_logs(config['round'], self.cid, loss, acc, model_size, train_time, cypher_time, decypher_time, self.dataset, self.solution)
        
        fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
        
        
        return [],len(self.x_train),fit_msg
    def evaluate(self, parameters, config):
        decypher_time = time.time()

        if len(config['he']) > 0:
            self.set_parameters(config)
        decypher_time = time.time() - decypher_time
        loss,acc = self.model.evaluate(self.x_test,self.y_test)

        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        write_evaluate_logs(config['round'], self.cid, loss, acc, decypher_time, self.dataset, self.solution)

        return loss, len(self.x_test), eval_msg
        
    def set_parameters(self,config):
        he_parameters        = ts.bfv_vector_from(self.context, config['he'])
        local_parameters     = self.model.get_weights()
        decrypted_parameters = he_parameters.decrypt()
        decrypted_parameters = unquantize(decrypted_parameters, 16, self.num_clients)
        if self.only_sum:
            decrypted_parameters  = np.array(decrypted_parameters) / float(config['total_examples'])
        
        reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
        self.model.set_weights(reshaped_parameters)