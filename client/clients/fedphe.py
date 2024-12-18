from base_client import BaseNumpyClient
import time 
import tenseal as ts 
import numpy as np
from client.common.utils import flat_parameters, cypher_packs,get_size,create_fit_msg, flat_packs, decypher_packs, remove_padding,reshape_parameters
from client.common.ranking import topk
from client.common.client_logs import write_evaluate_logs,write_train_logs
import pickle
def packing(parameters, chunk_size=4096):
    total_parameters = len(parameters)
    number_packs     = total_parameters // chunk_size

    last_chunk      = (number_packs) * chunk_size
    last_chunk_size = parameters[last_chunk:]
    need_to_fill    = chunk_size - last_chunk
    parameters.extend([0] * need_to_fill)

    packs            = []
    
    for pack in range(number_packs + 1):
        
        start  = pack * chunk_size
        end    = start + chunk_size
        packet = np.array(parameters[start : end])
        
        
        packs.append(packet)

    return packs
class FedPHEClient(BaseNumpyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager,
                 homomorphic_scheme = 'CKKS',only_sum = False):
        super().__init__(cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager)
        self. homomorphic_scheme = homomorphic_scheme
        self.solution = 'fedphe'

    def fit(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            if self.packing:
                self.set_parameter(self, config)    
        decypher_time = time.time() - decypher_time
        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        trained_parameters = self.model.get_weights()
        he_parameters      = []
        cypher_time        = time.time()
        flatted_parameters = flat_parameters(trained_parameters)     
        packed_parameters  = packing(flatted_parameters)
        
        if self.only_sum:
            packed_parameters = [np.array(pack) * len(self.x_train) for pack in packed_parameters] 
            # TODO: Refatorar para passar a função de seleção de mascara como parametro da classe    
            topk_mask          = topk(packed_parameters, 0.1)
            cyphered_packs     = cypher_packs(self, packed_parameters, topk_mask)
            he_parameters      = pickle.dumps(cyphered_packs)
            model_size         = get_size(cyphered_packs)
        
        cypher_time = time.time() - cypher_time
        
        write_train_logs(config['round'], self.cid, loss, acc, model_size, train_time, cypher_time, decypher_time, self.dataset, self.solution)
        
        fit_msg = create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
        
        return [],len(self.x_train),fit_msg
    def evaluate(self, parameters, config):
        decypher_time = time.time()
        if len(config['he']) > 0 :
            self.set_parameters(config)
        decypher_time = time.time() - decypher_time
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        write_evaluate_logs(config['round'], self.cid, loss, acc, decypher_time, self.dataset, self.solution)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        return loss, len(self.x_test), eval_msg

    def set_parameters(self,config):
        packed_parameters  = packing(flat_parameters(self.model.get_weights()))
        cyphered_packs     = pickle.loads(config['he'])
        decyphered_pack    = decypher_packs(cyphered_packs, self.context)
        aggredated_mask    = pickle.loads(config['mask'])
        
        for idx_mask, m in enumerate(aggredated_mask):
            if m > 0:
                if self.only_sum:
                    packed_parameters[idx_mask] = np.array(decyphered_pack.pop(0)) / m
                else:
                    packed_parameters[idx_mask] = decyphered_pack.pop(0)
        
        flatted_packs       = flat_packs(packed_parameters)
        flatted_packs       = remove_padding(self, flatted_packs)
        reshaped_parameters = reshape_parameters(flatted_packs,model=self.model)
        self.model.set_weights(reshaped_parameters)