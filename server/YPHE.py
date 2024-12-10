import flwr as fl
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate ,weighted_loss_avg
import pickle
import tenseal as ts
import os
import numpy as np
from functools import reduce
import logging
logging.basicConfig(level=logging.DEBUG)

import random

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



class YPHE_server(fl.server.strategy.FedAvg):
    def __init__(self, num_clients, dirichlet_alpha, dataset, fraction_fit=1.0):
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset         = dataset
        self.context         = self.get_server_context()
        self.agg_parameters  = []
        self.log_folder = get_latest_created_folder(LOG_DIR)

 
        super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, 
                         min_fit_clients=num_clients, min_evaluate_clients=num_clients)
        
    def get_server_context(self):
        with open(f'context/server_key.pkl', 'rb') as file:
            secret = pickle.load(file)  
        context = ts.context_from(secret["context"])
        return context
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        
        data2send = '' 
        print("CONFIGURE FIT")
        size_array_bytes = []
        if len(self.agg_parameters) > 0:
            data2send = self.agg_parameters
            
            for idx,layer in enumerate(data2send):
                print(layer)
                if layer != [0]:
                    data2send[idx] = layer.serialize()
                
                else:
                    data2send[idx] = pickle.dumps(layer)
                size_array_bytes.append(len(data2send[idx]))

        he =b''
        for b in data2send:
            he+=b
        config = {
            'he': he,
            "size_array_bytes":pickle.dumps(size_array_bytes)
        }
        
        fit_ins = FitIns(parameters, config)

		# Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
		    client_manager.num_available()
		)
        clients = client_manager.sample(
		    num_clients=sample_size, min_num_clients=min_num_clients
		)

		# Return client/config pairs
        print(clients)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):		
        weights_results = []
        agg_parameters  = 0
        parameters_list = []
        total_examples  = 0 
        mask = []
        for _, fit_res in results:
            print(f"RECEBIDO CLIENTE {fit_res.metrics['cid']}")
            client_id      = str(fit_res.metrics['cid'])
            mask = pickle.loads(fit_res.metrics["mask"])
            size_array_bytes = pickle.loads(fit_res.metrics["size_array_bytes"])

            he = fit_res.metrics['he']
            # print(len(he))
            # print(size_array_bytes)
            parameters = []
            idx = 0
            for size in size_array_bytes:
                parameters.append(he[idx:idx+size])
                idx = idx+size
            for idx,layer in enumerate(parameters):
                parameters[idx] = ts.ckks_vector_from(self.context,layer)
            

        
            matrix = [[0] for _ in mask]
            count_idx_parameter = 0
            for idx,row in enumerate(mask):
                if row == 1:
                    matrix[idx] = parameters[count_idx_parameter]
                    count_idx_parameter+=1
            # print(f"MATRIX:{matrix}")
            parameters_list.append((matrix, int(fit_res.num_examples)))
            total_examples  += int(fit_res.num_examples)
        idx = 0
        # print(f"AGG PARAMETERS: {self.agg_parameters}")
        agg_parameters = [[0] for _ in mask]
        print(f'AGG_PARAMETER GERADO MASCARA:{agg_parameters}')
        for parameters, num_examples in parameters_list:
            cl = []
            weights         = num_examples / total_examples
            for idx,layer in enumerate(parameters):
                if layer != [0] :
                    if agg_parameters[idx] == [0]:
                        agg_parameters[idx] = (layer * weights)
                    else:
                        agg_parameters[idx]  = agg_parameters[idx] + (layer * weights)
                # print(f"{idx} - {agg_parameters}")
                    # print(self.agg_parameters[idx] )
            #         cl.append(layer*weights)
            #     else:
            #         cl.append(layer)
            # if agg_parameters == []:
            #     agg_parameters= cl
            # else:
            #     for idx in range(len(agg_parameters)):
            #         agg_parameters[idx]+=cl[idx]
            print(f"agg_parameters:{agg_parameters}")
        print("\n\n\t\tFINAL DA AGREGAÇÃO")
        self.agg_parameters = agg_parameters
        print("AGG_PARAMETERS")
        print(self.agg_parameters)
        
        
        

        return [], {}
    
    # def aggregate(self, results):
    #     """Compute weighted average."""
    #     # Calculate the total number of examples used during training
    #     num_examples_total = sum([num_examples for _, num_examples in results])

    #     # Precompute the multiplicative inverse of num_examples_total
    #     inverse_num_examples_total = 1.0 / num_examples_total

    #     # Create a list of weights, each multiplied by the related number of examples
    #     weighted_weights = [
    #         [layer * num_examples for layer in weights] for weights, num_examples in results
    #     ]

    #     # Compute average weights of each layer using multiplication instead of division
    #     weights_prime = [
    #         reduce(np.add, layer_updates) * inverse_num_examples_total
    #         for layer_updates in zip(*weighted_weights)
    #     ]
    #     return weights_prime
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation losses using weighted average."""
        print("PARTE DA AGREGAÇÃO DA VALIDACAO SERVIDOR")
        print(f"FAILURES: {self.accept_failures},{failures}")
        print(f"RESULTS: {results}")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        accuracies = []

        for _, response in results:
            print(response)
            acc = response.metrics['accuracy']
            accuracies.append(acc)

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        print(f"ACURACIAS E LOSS:{accuracies},{loss_aggregated}")
        print(f"Round {server_round} aggregated loss: {loss_aggregated} aggregated accuracy: {sum(accuracies)/len(accuracies)}")
        with open(f'{self.log_folder}/server_evaluate.csv', 'a') as f:
            f.write(f"{sum(accuracies)/len(accuracies)},{loss_aggregated}\n")

        return loss_aggregated, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        data2send = '' 
        print("CONFIGURE EVALUATE")
        size_array_bytes = []
        # print(self.agg_parameters)
        if len(self.agg_parameters) > 0:
            data2send =  [x for x in self.agg_parameters]
            print("\n\n\t\tLOOP")
            for idx,layer in enumerate(data2send):
                
                print(data2send[idx])
                if layer != [0]:
                    data2send[idx] = data2send[idx].serialize()
                else:
                    data2send[idx] = pickle.dumps(layer)
                size_array_bytes.append(len(data2send[idx]))
        print("\n\n")
        he =b''
        for b in data2send:
            he+=b
        # he = data2send
        config = {
            'he': he,
            "size_array_bytes":pickle.dumps(size_array_bytes)
        }
        # Parameters and config
        # config = {
        #     'he': self.agg_parameters,
        # }  # {"server_round": server_round, "local_epochs": 1}

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        # Each pair of (ClientProxy, FitRes) constitutes a successful update from one of the previously selected clients
        return [(client, evaluate_ins) for client in clients]
