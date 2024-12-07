import tenseal as ts
import numpy as np
import pickle as pk

def align_packs(self, packed_parameters, mask):
    
    aligned_pack = [0 for i in range(len(mask))]
    
    for idx, m in enumerate(mask):
        if m == 1:
            aligned_pack[idx] = ts.ckks_vector_from(self.context, packed_parameters.pop(0))
            
    return aligned_pack

def aggregate_packs_and_masks(aligned_pack, weighted_mask, aggregated_packs, aggregated_mask):
    
    is_empty = len(aggregated_packs) == 0
    
    for i in range(len(weighted_mask)):
        if is_empty:
            print(weighted_mask[i])
    
            aggregated_packs.append(aligned_pack[i] * weighted_mask[i])
            aggregated_mask.append(weighted_mask[i])
        else:
            aggregated_packs[i] = aggregated_packs[i] + aligned_pack[i] * weighted_mask[i]
            aggregated_mask[i]  = aggregated_mask[i] + weighted_mask[i]
            
    return aggregated_packs, aggregated_mask

def serialize_packs(aggregated_packs):
    
    serialized_packs = []
    
    for pack in aggregated_packs:
        if pack != 0:
            serialized_packs.append(pack.serialize())
        
    return serialized_packs

def get_pack_mask_pair(self, results):
    pack_mask_pair   = []
    total_examples   = 0
    
    for _, fit_res in results:
        client_id         = str(fit_res.metrics['cid'])
        packed_parameters = pk.loads(fit_res.metrics['he'])
        mask              = pk.loads(fit_res.metrics['mask'])
        dataset_size      = int(fit_res.num_examples)
        
        aligned_pack    = align_packs(self, packed_parameters, mask)
        weighted_mask   = np.array(mask) * dataset_size
        total_examples += dataset_size  
        pack_mask_pair.append([aligned_pack, weighted_mask])
        
    return pack_mask_pair, total_examples

def aggregate_packs_and_mask(pack_mask_pair, total_examples):
    aggregated_packs = [0 for i in range(len(pack_mask_pair[0][1]))]
    aggregated_masks = [0 for i in range(len(pack_mask_pair[0][1]))]
    
    for packs, mask in pack_mask_pair:
        mask = mask / total_examples

        for idx_pack, p in enumerate(packs):
            aggregated_packs[idx_pack] = aggregated_packs[idx_pack] + (p * mask[idx_pack])
            aggregated_masks[idx_pack] = aggregated_masks[idx_pack] + mask[idx_pack]
            
    return aggregated_packs, aggregated_masks

def get_cyphered_parameters(self, results):
    parameters_list = []
    total_examples  = 0
    
    for _, fit_res in results:
        client_id      = str(fit_res.metrics['cid'])
        parameters     = ts.ckks_vector_from(self.context, fit_res.metrics['he']) 
        parameters_list.append((parameters, int(fit_res.num_examples)))
        total_examples  += int(fit_res.num_examples)

        #self.log_metrics_client(fit_res.metrics, server_round, end_delay)
    return parameters_list, total_examples
  

def aggregated_cyphered_parameters(parameters_list, total_examples):
    agg_parameters = 0
    
    for parameters, num_examples in parameters_list:
        weights         = num_examples / total_examples
        agg_parameters  = agg_parameters + (parameters * weights)
        
    return agg_parameters
    
    
