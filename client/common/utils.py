import time
import pickle
import tenseal as ts
import numpy as np
import sys
def flat_parameters(parameters):
    flat_params = []

    for param in parameters:
        flat_params.extend(param.flatten())

    return flat_params
def create_fit_msg(cid, train_time, acc, loss, model_size, he_parameters, mask=''):
    
    fit_msg = {
        'cid'         : cid,
        'train_time'  : train_time,
        'accuracy'    : acc,
        'loss'        : loss,
        'delay_start' : time.time(),
        'data_size'   : model_size,
        'he'          : he_parameters,
        'mask'        : pickle.dumps(mask)
    }
    
    return fit_msg
def get_client_context(homomorphic_scheme):
    file_path_scheme = {
        'Paillier':f'context/paillier.pkl',
        'CKKS' : f'../context/ckks_secret.pkl' ,
        'BFV': f'../context/bfv_secret.pkl'
    }
    with open(file_path_scheme[homomorphic_scheme],'rb') as file:
        secret = pickle.load(file)
        context = secret if homomorphic_scheme == 'Paillier' else ts.context_from(secret['context'])
    return context
def reshape_parameters(decrypted_parameters, model):
    reshaped_parameters = []

    for layer in model.get_weights():
        reshaped_parameters.append(np.reshape(decrypted_parameters[:layer.size], layer.shape))
        decrypted_parameters = decrypted_parameters[layer.size:]

    return reshaped_parameters
def cypher_packs(self, packs, mask):
    cyphered_packs   = []
    for idx, pack in enumerate(packs):        
        if mask[idx] == 1:
            
            cypher          = ts.ckks_vector(self.context, pack)
            serialized_pack = cypher.serialize()
            
            cyphered_packs.append(serialized_pack)

    return cyphered_packs
def get_size(data):
    total_size = 0
    
    for item in data:
        total_size += sys.getsizeof(item)
        
    return total_size

def decypher_packs(packed_parameters, context):
    decrypted_parameters = []
    
    for pack in packed_parameters:
        decrypted_parameters.append(ts.ckks_vector_from(context, pack).decrypt())
        
    return decrypted_parameters
def flat_packs(packed_parameters):
    flat_params = []
    
    for pack in packed_parameters:
        flat_params.extend(pack)
        
    return flat_params
def remove_padding(self, packed_parameters):
    total_parameters = len(flat_parameters(self.model.get_weights()))
    return packed_parameters[:total_parameters]