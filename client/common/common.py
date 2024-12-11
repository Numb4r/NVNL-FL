import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from pathlib import Path

def maior_divisor(n):
    if n <= 1:
        return None  # Não existe maior divisor para números <= 1

    for i in range(n // 2, 0, -1):  # Percorre de n/2 até 1
        if n % i == 0:  # Se i é divisor de n
            return i
def top_k(matrix,k,metric="norm"):
    layer_metrics = []
    for i in range(0,len(matrix),2):
        layer_metric = []
        if metric == "norm":
            score = np.linalg.norm(matrix[i])
        elif metric == "magnitude":
            score = np.sum(np.abs(matrix[i]))
        layer_metrics.append((i,score))
    layer_metrics.sort(key=lambda x:x[1],reverse=True)
    selected_layers = [layer_metrics[i][0] for i in range(min(k,len(layer_metrics)))]
    mask = [1 if idx in selected_layers else 0 for idx in range(len(matrix))]
    return selected_layers,mask
def load_data(niid,num_clients,dirichlet_alpha,dataset,cid):
    
    if niid:
        partitioner_train = DirichletPartitioner(num_partitions=num_clients, partition_by="label",
                                alpha=dirichlet_alpha, min_partition_size=100,
                                self_balancing=False)
    else:
        partitioner_train =  IidPartitioner(num_partitions=num_clients)
    
    fds               = FederatedDataset(dataset=dataset, partitioners={"train": partitioner_train})
    train             = fds.load_partition(cid).with_format("numpy")
    partitioner_test  = IidPartitioner(num_partitions=num_clients)
    fds_eval          = FederatedDataset(dataset=dataset, partitioners={"test": partitioner_test})
    test              = fds_eval.load_partition(cid).with_format("numpy")

    return train['image']/255.0, train['label'], test['image']/255.0, test['label']
def flat_parameters(parameters):
        flat_params = []

        for param in parameters:
            flat_params.extend(param.flatten())

        return flat_params

def reshape_parameters(decrypted_parameters,matrix):
    reshaped_parameters = []

    for c in matrix:
        c = np.array(c)
        reshaped_parameters.append(np.reshape(decrypted_parameters[:c.size], c.shape))
        decrypted_parameters = decrypted_parameters[c.size:]

    return reshaped_parameters
def get_latest_created_folder(directory):
    # Lista todos os diretórios dentro do caminho especificado
    folders = [f for f in Path(directory).iterdir() if f.is_dir()]
    
    # Se houver diretórios, ordene-os pela data de criação (metadata de criação - st_ctime)
    if folders:
        latest_folder = max(folders, key=lambda f: f.stat().st_ctime)
        return latest_folder
    else:
        return None