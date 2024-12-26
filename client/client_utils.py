import math
import sys
import tenseal as ts
import numpy as np
from models import flat_parameters

def get_size(data):
    total_size = 0
    
    for item in data:
        total_size += sys.getsizeof(item)
        
    return total_size
    
    
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

def cypher_packs(self, packs, mask):
    cyphered_packs   = []
    for idx, pack in enumerate(packs):        
        if mask[idx] == 1:
            
            cypher          = ts.ckks_vector(self.context, pack)
            serialized_pack = cypher.serialize()
            
            cyphered_packs.append(serialized_pack)

    return cyphered_packs

def get_topk_mask(packs, percentage):
    topk = int(np.ceil(len(packs) * percentage))
    
    avg_list     = [np.average(np.abs(pack)) for pack in packs]    
    max_avg_list = np.sort(avg_list)[::-1][:topk]
    mask_list    = []
    
    for i in range(len(max_avg_list)):
        mask_list.append(avg_list.index(max_avg_list[i]))
    mask_list.sort()
    
    res_mask = [0  for i in range(len(packs)) ]
    for i in range(len(packs)):
        if i in mask_list:
            res_mask[i] = 1
            
    return res_mask


def remove_padding(self, packed_parameters):
    total_parameters = len(flat_parameters(self.model.get_weights()))
    return packed_parameters[:total_parameters]

def flat_packs(packed_parameters):
    flat_params = []
    
    for pack in packed_parameters:
        flat_params.extend(pack)
        
    return flat_params

def decypher_packs(packed_parameters, context):
    decrypted_parameters = []
    
    for pack in packed_parameters:
        decrypted_parameters.append(ts.ckks_vector_from(context, pack).decrypt())
        
    return decrypted_parameters
def reshape_model(flatted_packs,a,sentinel=0):
    model =[]
    # print(a)
    
    if type(a) != int and type(a)!=float:
        # print(f'len:{len(a)}')
        for item in a:
            # print(f'item {item},  {type(item)},sentinel {sentinel}')
            
            m,sentinel=reshape_model(flatted_packs,item,sentinel)
            # print(sentinel)
            model.append(m)
    else:
    
        sentinel+=1 
        # print(f'a: {a}')
        return flatted_packs[sentinel-1],sentinel
    return model,sentinel

def packing_yphe(model_layers,chunk_size=-1):

    packs = []

    for idx,layer in enumerate(model_layers):
        weight = layer
        
        
        for w in weight:
            flat = flat_parameters(w)
            total_parameters = len(flat)
            current_chunk_size = chunk_size if chunk_size != -1 else total_parameters            
            number_packs = math.ceil(total_parameters / current_chunk_size )
            for pack in range(number_packs):
                start = pack * current_chunk_size
                end = start + current_chunk_size
                
            
                packs.append(flat[start:end])
            
            if len(packs[-1]) < chunk_size:
                packs[-1].extend([0] * (chunk_size - len(packs[-1])))
            
    return packs
def remove_padding_yphe(layer,target):
    
    total_parameters = len(target)
    return layer[:total_parameters]