import flwr as fl
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import tensorflow as tf
from tensorflow.keras import layers,models
import os
import random
import tenseal as ts
import pickle
import numpy as np
import time 
import sys
# logging.basicConfig(level=logging.DEBUG)

# import logging

from pathlib import Path

LOG_DIR = "logs/"
def residual_block(x, filters, kernel_size=3, stride=1):
    # Convolução 1
    shortcut = x  # A conexão de atalho
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Convolução 2
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Somar a entrada com a saída (conexão residual)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def get_latest_created_folder(directory):
    # Lista todos os diretórios dentro do caminho especificado
    folders = [f for f in Path(directory).iterdir() if f.is_dir()]
    
    # Se houver diretórios, ordene-os pela data de criação (metadata de criação - st_ctime)
    if folders:
        latest_folder = max(folders, key=lambda f: f.stat().st_ctime)
        return latest_folder
    else:
        return None


class HEClient(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, dirichlet_alpha):
        
        self.NOT_ENCRYPTED_LAYERS = 2
        self.log_folder = get_latest_created_folder(LOG_DIR)
        self.cid             = int(cid)
        self.dataset         = dataset
        self.niid            = niid
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.last_parameters = None                                   

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.model                                           = self.create_model(self.x_train.shape)
        self.context                                         = self.get_client_context()
        self.k = 5
        self.mask = []
        
    def get_client_context(self):
        with open(f'context/secret.pkl', 'rb') as file:
            secret = pickle.load(file)  

        context = ts.context_from(secret["context"])

        return context

    def get_parameters(self, config):
        parameters           = self.model.get_weights()
        return parameters
    
    def create_model(self, input_shape):
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Input(shape=(28, 28, 1)),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(32, activation='relu'),
        #     tf.keras.layers.Dense(16,  activation='relu'),
        #     tf.keras.layers.Dense(10, activation='softmax'),

        # ])
        model = tf.keras.models.Sequential([
           layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
           layers.MaxPooling2D((2, 2)),
           layers.Conv2D(64, (3, 3), activation='relu'),
           layers.MaxPooling2D((2, 2)),
           layers.Conv2D(128, (3, 3), activation='relu'),
           layers.MaxPooling2D((2, 2)),
           layers.Flatten(),
           layers.Dense(128, activation='relu'),
           layers.Dense(10, activation='softmax') 
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
        return model
 
    def top_k(self,matrix,k,metric="norm"):
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


    def load_data(self):
        
        if self.niid:
            partitioner_train = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=100,
                                    self_balancing=False)
        else:
            partitioner_train =  IidPartitioner(num_partitions=self.num_clients)
        
        fds               = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner_train})
        train             = fds.load_partition(self.cid).with_format("numpy")
        partitioner_test  = IidPartitioner(num_partitions=self.num_clients)
        fds_eval          = FederatedDataset(dataset=self.dataset, partitioners={"test": partitioner_test})
        test              = fds_eval.load_partition(self.cid).with_format("numpy")

        return train['image']/255.0, train['label'], test['image']/255.0, test['label']

    def flat_parameters(self, parameters):
        flat_params = []

        for param in parameters:
            flat_params.extend(param.flatten())

        return flat_params

    def reshape_parameters(self,decrypted_parameters,matrix):
        reshaped_parameters = []

        for c in matrix:
            c = np.array(c)
            reshaped_parameters.append(np.reshape(decrypted_parameters[:c.size], c.shape))
            decrypted_parameters = decrypted_parameters[c.size:]

        return reshaped_parameters
    # def reshape_parameters(self, decrypted_parameters):
    #     reshaped_parameters = []

    #     for layer in self.model.get_weights():
    #         reshaped_parameters.append(np.reshape(decrypted_parameters[:layer.size], layer.shape))
    #         decrypted_parameters = decrypted_parameters[layer.size:]

    #     return reshaped_parameters
    def set_parameters(self,parameters,config):
        
        if len(config['he']) > 0:
            he = config['he']
            size_arrays_bytes= pickle.loads(config['size_array_bytes'])
            local_parameters     = self.model.get_weights()
            he_parameters= [] 
            idx = 0
            for size in size_arrays_bytes:
                he_parameters.append(he[idx:idx+size])
                idx = idx+size

            # mask = config['mask']

            local_parameters=self.model.get_weights()
            for idx in range(len(self.mask)):
                if self.mask[idx] == 1:
                    decrypt_layer =ts.ckks_vector_from(self.context,he_parameters[idx]).decrypt()
                    reshaped = self.reshape_parameters(decrypt_layer,local_parameters[idx])
                    local_parameters[idx] = reshaped
                local_parameters[idx] = np.array(local_parameters[idx])
                    
            self.model.set_weights(local_parameters)
            # for idx,layer in he_parameters:
            #     if layer != [0]:
            #         decrypt_layer = ts.ckks_vector_from(self.context,layer)
            #         reshaped = self.reshape_parameters(decrypt_layer,local_parameters[idx])
            #         local_parameters[idx] = reshaped

            # for idx_decrypt,idx_local in enumerate(mask):
            #     reshaped = self.reshape_parameters(decrypted_parameters[idx_decrypt],local_parameters[idx_local])

            #     local_parameters[idx_local] = reshaped

        
            
            # self.model.set_weights(local_parameters)
        
        
    def fit(self, parameters, config):
        
        print(f"Client {self.cid} - {config.keys()} - {[len(x) for x in config.items()]}")
        tempo_cifragem_total = 0
        tempo_decrypt_total = 0 
        self.set_parameters(parameters,config)

        
        time_training = time.time()


        history = self.model.fit(self.x_train, self.y_train, epochs=1)
        time_training_total = time.time() - time_training
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        
        model_layers = []
        


        for idx,layer in enumerate(self.model.layers):    
            weight = layer.get_weights()
            if weight:
                flat_parameters = self.flat_parameters(weight[0])
                model_layers.append(flat_parameters) # pesos
                model_layers.append(weight[1]) # bias
        selected_layers,self.mask = self.top_k(model_layers,self.k,metric="norm")
        selected_layers = sorted(selected_layers)
        

        
        # trained_parameters = self.model.get_weights()
        
        send_layers = []
        for l in selected_layers:
            send_layers.append(model_layers[l])
        tamanho_flat_parameter = sys.getsizeof(send_layers) 

        size_array_bytes= []
        tempo_cifragem_inicio = time.time()
        for idx,l in enumerate(send_layers):
            send_layers[idx] = ts.ckks_vector(self.context,l).serialize()
            size_array_bytes.append(len(send_layers[idx]))
        print(size_array_bytes)
        tempo_cifragem_total = time.time() - tempo_cifragem_inicio
        

        
        tamanho_cifrado =      sum([sys.getsizeof(s) for s in send_layers])

        
        
        
        

        he = b''
        for b in send_layers:
            he+=b

        fit_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss,
            'he'      : he,
            'mask'     : pickle.dumps(self.mask),
            'size_array_bytes':pickle.dumps(size_array_bytes),
        }
        print(f"Selected layers {selected_layers}")
        print(f"Mask {self.mask}")
        print(f"acc : {acc}")
        print(f"loss : {loss}")
        print(f"tamanho he: {len(he)}")
        print(f"tamanho_flat_parameter : {tamanho_flat_parameter}")
        print(f"tamanho_cifrado : {tamanho_cifrado}")
        print(f"tempo_cifragem_total : {tempo_cifragem_total}")
        print(f"tempo_decrypt_total : {tempo_decrypt_total}")
        print(f"time_training_tota : {time_training_total}")
        
        with open(f'{self.log_folder}/client_{self.cid}_train.csv', 'a') as f:
            f.write(f"{acc},{loss},{tamanho_flat_parameter},{tamanho_cifrado},{tempo_cifragem_total},{tempo_decrypt_total},{time_training_total} \n")
        return [], len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        
        print("VALIDAÇÃO")
        if len(config['he']) > 0:
            
        
            he = config['he']
            size_arrays_bytes= pickle.loads(config['size_array_bytes'])
            local_parameters     = self.model.get_weights()
            he_parameters= [] 
            idx = 0
            for size in size_arrays_bytes:
                he_parameters.append(he[idx:idx+size])
                idx = idx+size


            # tempo_decrypt_inicio = time.time()
            # decrypted_parameters = he_parameters.decrypt()
            # tempo_decrypt_total = time.time() - tempo_decrypt_inicio
            # mask = config['mask']

            local_parameters=self.model.get_weights()
            for idx in range(len(self.mask)):
                if self.mask[idx] == 1:
                    decrypt_layer = ts.ckks_vector_from(self.context,he_parameters[idx]).decrypt()
                    reshaped = self.reshape_parameters(decrypt_layer,local_parameters[idx])
                    local_parameters[idx] = reshaped
                local_parameters[idx] = np.array(local_parameters[idx])
            self.model.set_weights(local_parameters)

        # ------------------------------------
        #     he_parameters        = ts.ckks_vector_from(client_context, config['he'])
        #     local_parameters     = self.model.get_weights()
        #     temp_flat            = self.flat_parameters(local_parameters[:self.NOT_ENCRYPTED_LAYERS])
        #     decrypted_parameters = he_parameters.decrypt()
        #     temp_flat.extend(decrypted_parameters)
            
        #     reshaped_parameters  = self.reshape_parameters(temp_flat)
        #     self.model.set_weights(reshaped_parameters)

        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }
        with open(f'{self.log_folder}/client_{self.cid}_eval.csv', 'a') as f:
            f.write(f"{acc},{loss}\n")
        

        return loss, len(self.x_test), eval_msg
    

def main():
	
	client =  HEClient(
					cid             = int(os.environ['CID']), 
                    niid            = bool(os.environ['NIID']), 
                    dataset         = os.environ['DATASET'], 
                    num_clients     = int(os.environ['NCLIENTS']), 
                    dirichlet_alpha = float(os.environ['DIRICHLET_ALPHA'])
                    )
	
	fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                              client=client)


if __name__ == '__main__':
	main()