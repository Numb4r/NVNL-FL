import flwr as fl
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import tensorflow as tf
import os
import random
import tenseal as ts
import pickle
from dataset_utils import ManageDatasets
import tracemalloc
import sys
import pickle
import time
# import logging
# logging.basicConfig(level=logging.DEBUG)

class HEClient(fl.client.NumPyClient):
    def __init__(self, cid, niid, dataset, num_clients, 
                 dirichlet_alpha, start2share, homomorphic, homomorphic_type):
        
        self.cid              = int(cid)
        self.dataset          = dataset
        self.niid             = niid
        self.num_clients      = num_clients
        self.dirichlet_alpha  = dirichlet_alpha
        self.last_parameters  = None
        self.dataset_manager  = ManageDatasets(self.cid)
        self.start2share      = start2share
        self.homomorphic      = homomorphic
        self.homomorphic_type = homomorphic_type                                    

        if dataset == 'MNIST' or dataset == 'CIFAR10':
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        else:
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_har(dataset) #self.load_data()
            

        if dataset == 'CIFAR10':
            self.model                                               = self.create_ccn(self.x_train.shape, len(np.unique(self.y_train)))

        else:
            self.model                                               = self.create_model(self.x_train.shape, len(np.unique(self.y_train)))
        
        if self.homomorphic:
            self.context = self.get_client_context()
        
        
    def get_client_context(self):
        if self.homomorphic_type == 'Full':
            with open(f'../context/secret.pkl', 'rb') as file:
                secret = pickle.load(file)
        else:
            with open(f'../context/secret_partial.pkl', 'rb') as file:
                secret = pickle.load(file)  

        context = ts.context_from(secret["context"])

        return context

    def get_parameters(self, config):
        parameters           = self.model.get_weights()
        return parameters
    
    def load_har(self, dataset):
        if dataset == 'UCIHAR':
            return self.dataset_manager.load_UCIHAR()
        if dataset == 'ExtraSensory':
            return self.dataset_manager.load_ExtraSensory()
        if dataset == 'MotionSense':
            return self.dataset_manager.load_MotionSense()
    
    def create_model(self, input_shape, output):

        print(f"Creating model with input shape {input_shape} and output {output}")

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_shape[1:])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64,  activation='relu'),
            tf.keras.layers.Dense(32,  activation='relu'),
            tf.keras.layers.Dense(output, activation='softmax'),

        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
        return model
    
    def create_ccn(self, input_shape, num_classes):
        model = tf.keras.models.Sequential()

		# Convolutional Block 1
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape[1:]))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        # Convolutional Block 2
        # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(tf.keras.layers.Dropout(0.3))

        # # Convolutional Block 3
        # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(tf.keras.layers.Dropout(0.4))

        # Fully Connected Layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

        if self.dataset == 'CIFAR10':
            return train['img']/255.0, train['label'], test['img']/255.0, test['label']
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
        
        decyfer_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            he_parameters        = ts.ckks_vector_from(self.context, config['he'])
            local_parameters     = self.model.get_weights()
            # temp_flat            = self.flat_parameters(local_parameters[:self.start2share])
            decrypted_parameters = he_parameters.decrypt()
            # temp_flat.extend(decrypted_parameters)
            
            reshaped_parameters  = self.reshape_parameters(decrypted_parameters)
            self.model.set_weights(reshaped_parameters)
        
        else:
            self.model.set_weights(parameters)
        decyfer_time = time.time() - decyfer_time

        train_time = time.time()
        history    = self.model.fit(self.x_train, self.y_train, epochs=1)
        train_time = time.time() - train_time

        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])

        
        trained_parameters = self.model.get_weights()
        he_parameters      = []
        cyfer_time         = time.time()
        if self.homomorphic:
            flat_parameters    = self.flat_parameters(trained_parameters) 
            he_parameters      = ts.ckks_vector(self.context, flat_parameters)
            he_parameters      = he_parameters.serialize()
            model_size         = sys.getsizeof(he_parameters) 
        
        else:
            flat_parameters    = self.flat_parameters(trained_parameters)
            temp_buf   = pickle.dumps(flat_parameters)
            model_size = sys.getsizeof(temp_buf)
        
        cyfer_time = time.time() - cyfer_time
        
        # filename = f'logs/{self.dataset}/train_fhe_{self.start2share}_{self.cid}.csv' if self.homomorphic else f'logs/{self.dataset}/train_{self.cid}.csv'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)

        # with open(filename, 'a') as file:
        #     file.write(f"{config['round']}, {self.cid}, {loss}, {acc}, {model_size}, {train_time}, {cyfer_time}, {decyfer_time}\n")

        fit_msg = {
            'cid'         : self.cid,
            'train_time'  : train_time,
            'accuracy'    : acc,
            'loss'        : loss,
            'delay_start' : time.time(),
            'data_size'   : model_size,
            'he'          : he_parameters if self.homomorphic else ''
        }

        return trained_parameters, len(self.x_train), fit_msg

    def evaluate(self, parameters, config):
        decyfer_time = time.time()
        if len(config['he']) > 0 and self.homomorphic:
            client_context = self.get_client_context()
            he_parameters        = ts.ckks_vector_from(client_context, config['he'])
            # local_parameters     = self.model.get_weights()
            # temp_flat            = self.flat_parameters(local_parameters)
            decrypted_parameters = he_parameters.decrypt()
            # temp_flat.extend(decrypted_parameters)
            
            reshaped_parameters  = self.reshape_parameters(decrypted_parameters)
            # print(reshaped_parameters)
            self.model.set_weights(reshaped_parameters)
        else:
            self.model.set_weights(parameters)
        decyfer_time = time.time() - decyfer_time
        loss, acc = self.model.evaluate(self.x_test, self.y_test)

        # filename = f'logs/{self.dataset}/evaluate_fhe_{self.start2share}_{self.cid}.csv' if self.homomorphic else f'logs/{self.dataset}/evaluate_{self.cid}.csv'
        # os.makedirs(os.path.dirname(filename), exist_ok=True)

        # with open(filename, 'a') as file:
        #     file.write(f"{config['round']}, {self.cid}, {loss}, {acc}, {decyfer_time}\n")

        eval_msg = {
            'cid'     : self.cid,
            'accuracy': acc,
            'loss'    : loss
        }

        return loss, len(self.x_test), eval_msg
    

def main():
    
    client =  HEClient(
                        cid             = int(os.environ['CID']), 
                        niid            = os.environ['NIID'] == 'True', 
                        dataset         = os.environ['DATASET'], 
                        num_clients     = int(os.environ['NCLIENTS']), 
                        dirichlet_alpha = float(os.environ['DIRICHLET_ALPHA']),
                        start2share     = int(os.environ['START2SHARE']),
                        homomorphic     = os.environ['HOMOMORPHIC'] == 'True',
                        homomorphic_type= str(os.environ['HOMOMORPHIC_TYPE'])
                        )
        
    fl.client.start_numpy_client(server_address=os.environ['SERVER_IP'], 
                                client=client)


if __name__ == '__main__':
	main()