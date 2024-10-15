import flwr as fl
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import tensorflow as tf
import os
import random
import tenseal as ts
import pickle
import numpy as np
# logging.basicConfig(level=logging.DEBUG)

# import logging

from pathlib import Path

LOG_DIR = "logs/"


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
        
        
    def get_client_context(self):
        with open(f'context/secret.pkl', 'rb') as file:
            secret = pickle.load(file)  

        context = ts.context_from(secret["context"])

        return context

    def get_parameters(self, config):
        parameters           = self.model.get_weights()
        return parameters
    
    def create_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16,  activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),

        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
        return model

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

    def reshape_parameters(self, decrypted_parameters):
        reshaped_parameters = []

        for layer in self.model.get_weights():
            reshaped_parameters.append(np.reshape(decrypted_parameters[:layer.size], layer.shape))
            decrypted_parameters = decrypted_parameters[layer.size:]

        return reshaped_parameters

    def fit(self, parameters, config):
        
        print(f"Client {self.cid} - {len(self.x_train)} - {config['total_samples']} ")
        
        if len(config['he']) > 0:
            he_parameters        = ts.ckks_tensor_from(self.context, config['he'])
            local_parameters     = self.model.get_weights()
            temp_flat            = self.flat_parameters(local_parameters[:self.NOT_ENCRYPTED_LAYERS])
            decrypted_parameters = he_parameters.decrypt().raw
            temp_flat.extend(decrypted_parameters)
            temp_flat = np.array(temp_flat)
            total_samples = config['total_samples']
            temp_flat /= total_samples
            
            reshaped_parameters  = self.reshape_parameters(temp_flat)
            self.model.set_weights(reshaped_parameters)

        #parameters_decoded = self.encoder.decrypt_decode_double(parameters)
        # self.model.set_weights(self.last_parameters)

        history = self.model.fit(self.x_train, self.y_train, epochs=1)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        
        trained_parameters = self.model.get_weights()
        # print(np.array(trained_parameters).shape())
        # trained_parameters = list(map(lambda x : list(map(lambda y : y*len(self.x_train),x)),trained_parameters))
        flat_parameters    = self.flat_parameters(trained_parameters[self.NOT_ENCRYPTED_LAYERS:])
        # flat_parameters = list(map(lambda x: x*self.x_train,flat_parameters))
        flat_parameters = np.array(flat_parameters)
        flat_parameters *= len(self.x_train) 
        # he_parameters      = ts.ckks_tensor(client_context, trained_parameters[-1]) 
        he_parameters      = ts.ckks_tensor(self.context, flat_parameters) 
        serialized = he_parameters.serialize()

        fit_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss,
            'he'      : serialized
        }
        
        with open(f'{self.log_folder}/client_{self.cid}_train.csv', 'a') as f:
            f.write(f"{acc},{loss},{len(flat_parameters)},{len(serialized)} \n")
        return self.flat_parameters(trained_parameters[:self.NOT_ENCRYPTED_LAYERS]), len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        client_context = self.get_client_context()
        
        if len(config['he']) > 0:
            he_parameters        = ts.ckks_tensor_from(client_context, config['he'])
            local_parameters     = self.model.get_weights()
            temp_flat            = self.flat_parameters(local_parameters[:self.NOT_ENCRYPTED_LAYERS])
            decrypted_parameters = he_parameters.decrypt().raw
            temp_flat.extend(decrypted_parameters)
            
            temp_flat = np.array(temp_flat)
            total_samples = config['total_samples']
            temp_flat /= total_samples
            
            reshaped_parameters  = self.reshape_parameters(temp_flat)
            self.model.set_weights(reshaped_parameters)

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