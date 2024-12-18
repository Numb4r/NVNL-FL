import time
import numpy as np
import math
import sys
import pickle
from base_client import BaseNumpyClient
from client.common.utils import flat_parameters, cypher_packs,get_size,create_fit_msg, flat_packs, decypher_packs, remove_padding,reshape_parameters
from client.common.client_logs import write_evaluate_logs,write_train_logs
def packing(model_layers,chunk_size=-1):
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
def remove_padding(layer,target):
    
    total_parameters = len(target)
    return layer[:total_parameters]
def remove_padding(layer,target):
    total_parameters = len(target)
    return layer[:total_parameters]

class YPHEClient(BaseNumpyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager,chunk_size=-1,only_sum=False):
        super().__init__(cid, niid, dataset, num_clients, dirichlet_alpha, dataset_magager)
        self.solution = 'yphe'
        self.only_sum = only_sum
        self.chunk_size=chunk_size
    def fit(self, parameters, config):
        fit_msg = create_fit_msg()
        write_train_logs(config['round'], self.cid, loss, acc, model_size, train_time, cypher_time, decypher_time, self.dataset, self.solution)

        return [],len(self.x_train), fit_msg
    
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
        return loss,len(self.x_test),eval_msg
    def set_parameters(self,config):
        packed_parameters  = packing(self.model.get_weights(),self.chunk_size)
        cyphered_packs     = pickle.loads(config['he'])
        decyphered_pack    = decypher_packs(cyphered_packs, self.context)
        aggredated_mask    = pickle.loads(config['mask'])
        
        for idx_mask, m in enumerate(aggredated_mask):
            
            if m > 0:
                if self.only_sum:
                    packed_parameters[idx_mask] = np.array(decyphered_pack.pop(0)) / m
                else:
                    packed_parameters[idx_mask] = decyphered_pack.pop(0)
        
                
                packed_parameters[idx_mask] = remove_padding(packed_parameters[idx_mask],weight[idx_mask])


        flatted_packs       = flat_packs(packed_parameters)
        reshaped_parameters =[]
        for idx,layer in enumerate(self.model.layers):
            # print(layer)
            cut = len(flat_parameters(layer))
            # print(f'cut {cut}')
            reshaped_parameters.append(reshape_parameters(flatted_packs[idx*cut:idx*cut+cut],layer))
        self.model.set_weights(reshaped_parameters)


# he = config['he']
# size_arrays_bytes= pickle.loads(config['size_array_bytes'])
# local_parameters     = self.model.get_weights()
# he_parameters= [] 
# idx = 0
# for size in size_arrays_bytes:
#     he_parameters.append(he[idx:idx+size])
#     idx = idx+size

# # mask = config['mask']

# local_parameters=self.model.get_weights()
# for idx in range(len(self.mask)):
#     if self.mask[idx] == 1:
#         decrypt_layer =ts.ckks_vector_from(self.context,he_parameters[idx]).decrypt()
#         reshaped = reshape_parameters(decrypt_layer,local_parameters[idx])
#         local_parameters[idx] = reshaped
#     local_parameters[idx] = np.array(local_parameters[idx])
        
# self.model.set_weights(local_parameters)
# # ===================================================
# # ===================================================
# # ===================================================
# # ===================================================



# import numpy as np
# import math
# a = np.array( [
#                     [[
#                         [1,2,3],
#                         [4,5,6]
#                     ],
#                     [7,8,9]],
#                     [[
#                         [1,2,3],
#                         [4,5,6]
#                     ],
#                     [7,8,9]]
#                 ],dtype='object')
# b = np.array( [
#                     [[
#                         [10,20,30],
#                         [40,50,60]
#                     ],
#                     [70,80,90]],
#                     [[
#                         [10,20,30],
#                         [40,50,60]
#                     ],
#                     [70,80,90]]
#                 ],dtype='object')      
# def flat_packs(packed_parameters):
#     flat_params = []
    
#     for pack in packed_parameters:
#         flat_params.extend(pack)
        
#     return flat_params
# def remove_padding(layer,target):
#     print(f'layer:{layer}')
#     print(f'target:{target}')
#     total_parameters = len(target)
#     return layer[:total_parameters]
# def reshape_parameters(decrypted_parameters, model):
#     reshaped_parameters = []

#     for layer in model:
#         layer = np.array(layer)
#         reshaped_parameters.append(np.reshape(decrypted_parameters[:len(layer)], layer.shape))
#         decrypted_parameters = decrypted_parameters[layer.size:]

#     return reshaped_parameters
# def flat_parameters(parameters):
#     flat_params = []

#     for param in parameters:
#         param = np.array(param)
#         flat_params.extend(param.flatten())
#     return flat_params
# def packing(model_layers,chunk_size=-1):
#     packs = []
    
#     for idx,layer in enumerate(model_layers):
#         weight = layer
#         # print(weight)
        
#         for w in weight:
#             flat = flat_parameters(w)
#             total_parameters = len(flat)
#             current_chunk_size = chunk_size if chunk_size != -1 else total_parameters            
#             number_packs = math.ceil(total_parameters / current_chunk_size )
#             for pack in range(number_packs):
#                 start = pack * current_chunk_size
#                 end = start + current_chunk_size
                
            
#                 packs.append(flat[start:end])
            
#             if len(packs[-1]) < chunk_size:
#                 packs[-1].extend([0] * (chunk_size - len(packs[-1])))
            
#     return packs

# packed_parameters   = packing(a,10)
# aggredated_mask     = [0,0,1,0]
# decyphered_pack     = packing(b.tolist(),10)


# weight = packing(a,-1)
# for idx_mask, m in enumerate(aggredated_mask):
#     # print(idx_mask,idx_mask//2)
#     # print(a.shape)
#     # print(a[idx_mask//2])
#     if m > 0:
#     #     if self.only_sum:
#     #         packed_parameters[idx_mask] = np.array(decyphered_pack.pop(0)) / m
#     #     else:
#         packed_parameters[idx_mask] = decyphered_pack.pop(0)
#     packed_parameters[idx_mask] = remove_padding(packed_parameters[idx_mask],weight[idx_mask])

# print(packed_parameters)
# flatted_packs       = flat_packs(packed_parameters)
# reshaped_parameters = reshape_parameters(flatted_packs,model=a)
# print(reshaped_parameters)



