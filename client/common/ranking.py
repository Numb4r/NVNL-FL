import numpy as np
def topk(packs, percentage):
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