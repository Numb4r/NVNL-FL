import time
import pickle 
import numpy as np
import sys
import tenseal as ts


from models import flat_parameters, reshape_parameters
from client_logs import write_train_logs, write_evaluate_logs
from client_utils import get_size, packing, cypher_packs, get_topk_mask, decypher_packs, flat_packs, remove_padding,get_pondering_random_mask,get_robin_round_mask,get_slice_window_mask

from encryption.quantize import quantize, unquantize, batch_padding, unbatching_padding

def he_parameters_to_model(self, config):
        
        if self.homomorphic_type == 'Paillier':
            quan_bits     = 32
            batch_size    = 50
            padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
            elem_bits     = quan_bits + padding_bits
            
            he_parameters = pickle.loads(config['he'])
            he_parameters = self.context.decrypt(he_parameters)
            he_parameters = unbatching_padding(he_parameters, elem_bits, batch_size)[:(int(np.prod((self.len_shared_data))))]
            he_parameters = unquantize(he_parameters, quan_bits, self.num_clients)
            if self.only_sum:
                he_parameters = np.array(he_parameters) / float(config['total_examples']) 
            
            print(f'HE PARAMS: {he_parameters[:20]}')
            
            reshaped_parameters = reshape_parameters(self, he_parameters)
            self.model.set_weights(reshaped_parameters)
        
        else:
            if 'CKKS' == self.homomorphic_type:
                he_parameters        = ts.ckks_vector_from(self.context, config['he'])
                local_parameters     = self.model.get_weights()
                decrypted_parameters = he_parameters.decrypt()
                if self.only_sum:
                    decrypted_parameters  = np.array(decrypted_parameters) / float(config['total_examples'])
                    
                reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
                self.model.set_weights(reshaped_parameters)
                
            else:
                he_parameters        = ts.bfv_vector_from(self.context, config['he'])
                local_parameters     = self.model.get_weights()
                decrypted_parameters = he_parameters.decrypt()
                decrypted_parameters = unquantize(decrypted_parameters, 16, self.num_clients)
                if self.only_sum:
                    decrypted_parameters  = np.array(decrypted_parameters) / float(config['total_examples'])
                
                reshaped_parameters  = reshape_parameters(self, decrypted_parameters)
                self.model.set_weights(reshaped_parameters)
            
def he_packs_to_model(self, config):
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
    reshaped_parameters = reshape_parameters(self, flatted_packs)
    self.model.set_weights(reshaped_parameters)


def fit_ckks(self, parameters, config):
    print(f'tenseal: {self.context}')
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
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
            flatted_parameters = np.array(flatted_parameters) * len(self.x_train)
            
        he_parameters      = ts.ckks_vector(self.context, flatted_parameters)
        he_parameters      = he_parameters.serialize()
        model_size         = sys.getsizeof(he_parameters)
        topk_mask          =  '' 
    
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    
    return fit_msg


def fit_bfv(self, parameters, config):
    print(f'tenseal: {self.context}')
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
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
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    
    return fit_msg
    
def fit_batchcrypt(self, parameters, config):
    
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        he_parameters_to_model(self, config)
    decypher_time = time.time() - decypher_time

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time         = time.time()
    
    
    if self.homomorphic and not self.packing:
        flatted_parameters = flat_parameters(trained_parameters) 
        flatted_parameters = np.array(flatted_parameters) #* len(self.x_train)
    
        quan_bits     = 32
        batch_size    = 50
        padding_bits  = int(np.ceil(np.log2(self.num_clients + 1)))
        elem_bits     = quan_bits + padding_bits
            
        quan_param    = quantize(flatted_parameters, quan_bits, self.num_clients)
        quan_param    = batch_padding(quan_param, self.context.key_length, elem_bits, batch_size=batch_size)

        he_parameters = self.context.encrypt(quan_param)
        print(f'HE PARAMS antes dumps: {sys.getsizeof(he_parameters)}')
        he_parameters = pickle.dumps(he_parameters)
        model_size    = sys.getsizeof(he_parameters)
        print(f'HE PARAMS depois dumps: {sys.getsizeof(he_parameters)}')
        topk_mask     =  '' 
        self.len_shared_data = len(flatted_parameters)
        
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    
    return fit_msg

def fit_fedphe(self, parameters, config):
   
    decypher_time = time.time()
    if len(config['he']) > 0 and self.homomorphic:
        if self.packing:
            he_packs_to_model(self, config)    
    decypher_time = time.time() - decypher_time

    train_time = time.time()
    history    = self.model.fit(self.x_train, self.y_train, epochs=1)
    train_time = time.time() - train_time

    acc     = np.mean(history.history['accuracy'])
    loss    = np.mean(history.history['loss'])

    trained_parameters = self.model.get_weights()
    he_parameters      = []
    cypher_time        = time.time()
    
    if self.packing:
        flatted_parameters = flat_parameters(trained_parameters)     
        packed_parameters  = packing(flatted_parameters)
        
        if self.only_sum:
            packed_parameters = [np.array(pack) * len(self.x_train) for pack in packed_parameters] 
            
        # topk_mask          = get_topk_mask(packed_parameters, 0.8)
        # topk_mask          = get_robin_round_mask(round=config['round'],packs=packed_parameters,size_window=0.8,percentage=True)
        # topk_mask          = get_slice_window_mask(round=config['round'],packs=packed_parameters,size_window=0.8,stride=0.8,percentage=True)

        if len(self.weights_packs ) == 0 :
            self.weights_packs = np.ones(len(packed_parameters))
        topk_mask          = get_pondering_random_mask(packs=packed_parameters,k=0.1,weights_packs=self.weights_packs,percentage=True)

        cyphered_packs     = cypher_packs(self, packed_parameters, topk_mask)
        he_parameters      = pickle.dumps(cyphered_packs)
        model_size         = get_size(cyphered_packs)
    
    cypher_time = time.time() - cypher_time
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
    
    return fit_msg


def fit_plaintext(self, parameters, config):
    
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
    
    write_train_logs(self, config['round'], loss, acc, model_size, train_time, cypher_time, decypher_time)
    
    fit_msg = self.create_fit_msg(train_time, acc, loss, model_size, he_parameters, topk_mask)
       
    return fit_msg


