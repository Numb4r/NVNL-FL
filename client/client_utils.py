import sys
import tenseal as ts
import numpy as np
from models import flat_parameters
import math
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
def get_slice_window_mask(round,packs,size_window=0.1,stride=3,percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
        stride = math.ceil(size_window*stride)
    
    array_z = np.zeros(len(packs))
    start = round*stride
    end = round*stride+size_window
    for i in range(start, end):
        array_z[i % len(array_z)] = 1
    return array_z

def get_robin_round_mask(round,packs,size_window,percentage=False):
    if percentage:
        size_window = math.ceil(len(packs) * size_window)
    return get_slice_window_mask(round,packs,size_window,size_window)
def get_pondering_random_mask(packs,k,weights_packs,penalty=0.5,reward=0.1,percentage=False):

    n = len(packs)
    if percentage:
        k = math.ceil(k*n)
    probabilities = weights_packs / weights_packs.sum()
    mask = np.zeros(n)

    
    chosen = np.random.choice(range(n), size=k, replace=False, p=probabilities)
    
    for i in range(n):
        if i in chosen:
            weights_packs[i] *= penalty  
            mask[i] = 1
        else:
            weights_packs[i] += reward  # Aumenta o peso dos não escolhidos

    
    weights_packs = np.maximum(weights_packs, 0.01)
    return mask



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