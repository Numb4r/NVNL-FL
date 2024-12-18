from base_client import BaseNumpyClient
from client.common.utils import get_client_context,reshape_parameters,flat_parameters
from client.common.client_logs import write_evaluate_logs,write_train_logs
import time 
import numpy as np
import pickle
import tenseal as ts
import sys
from client.encryption.quantize import unbatching_padding,unquantize
class BatchcryptClient(BaseNumpyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager,
                  homomorphic_scheme = 'Paillier',quan_bits = 32, batch_size = 50):
        super().__init__(cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager)
        self.context = get_client_context(homomorphic_scheme)
        self.solution = 'batchcrypt'
        self.quan_bits = quan_bits 
        self.batch_size = batch_size
   

    def fit(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0:
            self.set_parameters(self, config)
        decypher_time = time.time() - decypher_time

        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()
        he_parameters      = []
        cypher_time         = time.time()
        flatted_parameters = flat_parameters(trained_parameters) 
        
        if self.only_sum:
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
            
        he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
        he_parameters      = he_parameters.serialize()
        model_size         = sys.getsizeof(he_parameters)
        topk_mask          =  '' 
    
        cypher_time = time.time() - cypher_time
        
        write_train_logs(config['round'], self.cid, loss, acc, model_size, train_time, cypher_time, decypher_time, self.dataset, self.solution)
        
        fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
        
        return [],len(self.x_train),fit_msg
    def evaluate(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 :
            self.set_parameters(config)
        decypher_time = time.time() - decypher_time
        loss,acc = self.model.evaluate(self.x_test,self.y_test)
        write_evaluate_logs(config['round'], self.cid, loss, acc, decypher_time, self.dataset, self.solution)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        return loss, len(self.x_test), eval_msg
    

    def set_parameters(self,config):
        
        padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
        elem_bits     = self.quan_bits + padding_bits
        
        he_parameters = pickle.loads(config['he'])
        he_parameters = self.context.decrypt(he_parameters)
        he_parameters = unbatching_padding(he_parameters, elem_bits, self.batch_size)[:(int(np.prod((self.len_shared_data))))]
        he_parameters = unquantize(he_parameters, self.quan_bits, self.num_clients)
        if self.only_sum:
            he_parameters = np.array(he_parameters) / float(config['total_examples']) 
        
        print(f'HE PARAMS: {he_parameters[:20]}')
        
        reshaped_parameters = reshape_parameters(self, he_parameters)
        self.model.set_weights(reshaped_parameters)