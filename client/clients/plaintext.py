from base_client import BaseNumpyClient
import time
import numpy as np
from client.common.utils import flat_parameters,create_fit_msg
import sys
from client.common.client_logs import write_evaluate_logs,write_train_logs
class PlainTextClient(BaseNumpyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager):
        super().__init__(cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager)
        self.solution = 'plaintext'
    def fit(self, parameters, config):

        decypher_time = 0
        self.model.set_weights(parameters)
        

        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()
        he_parameters      = []
        
        
        flatted_parameters  = flat_parameters(trained_parameters)
        # temp_buf            = pickle.dumps(flatted_parameters)
        model_size          = sys.getsizeof(flatted_parameters)
        topk_mask           =  '' 
        cypher_time         = 0
        
        write_train_logs(config['round'], self.cid, loss, acc, model_size, train_time, cypher_time, 0, self.dataset, self.solution)
        
        fit_msg = create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
        
        return [],len(self.x_train),fit_msg
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss,acc = self.model.evaluate(self.x_test,self.y_test)
        write_evaluate_logs(config['round'], self.cid, loss, acc, 0, self.dataset, self.solution)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        return loss, len(self.x_test), eval_msg