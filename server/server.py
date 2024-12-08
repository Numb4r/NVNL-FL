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
import time
import random

from server_utils import align_packs, aggregate_packs_and_masks, serialize_packs, get_pack_mask_pair 
from server_utils import aggregate_packs_and_mask, get_cyphered_parameters, aggregated_cyphered_parameters

import pickle as pk
from encryption.paillier import PaillierCipher

class HEServer(fl.server.strategy.FedAvg):
    def __init__(self, num_clients, dirichlet_alpha, dataset, fraction_fit, homomorphic, packing, onlysum, homomorphic_type):
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset         = dataset
        self.agg_parameters  = ''
        self.agg_mask        = ''
        self.homomorphic     = homomorphic
        self.packing         = packing
        self.onlysum         = onlysum
        self.selection_time  = 0
        self.total_examples  = 0
        self.homomorphic_type= homomorphic_type
        self.context         = self.get_server_context()
        
        super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, min_evaluate_clients=num_clients)
        
    def get_server_context(self):
        
        if self.homomorphic_type == 'Paillier':
            with open(f'context/paillier.pkl', 'rb') as file:
                context = pickle.load(file)    
        
        else:
            with open(f'context/server_key.pkl', 'rb') as file:
                secret = pickle.load(file)  
                context = ts.context_from(secret["context"])
        
        return context
    
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        selection_start = time.time()
        data2send = '' 
        if len(self.agg_parameters) > 0:
            data2send = self.agg_parameters
        
        config = {
            'he': data2send,
            'mask': self.agg_mask,
            'round' : server_round,
            'total_examples' : str(self.total_examples),
        }
        
        fit_ins = FitIns(parameters, config)

		# Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
		    client_manager.num_available(), 
		)
        clients = client_manager.sample(
		    num_clients=sample_size, min_num_clients=min_num_clients
		)

		# Return client/config pairs
        self.selection_time = time.time() - selection_start

        return [(client, fit_ins) for client in clients]
    
    def log_metrics_client(self, metrics, server_round, end_delay):
        cid         = int(metrics['cid'])
        train_time  = float(metrics['train_time'])
        delay       = end_delay - float(metrics['delay_start'])
        loss        = float(metrics['loss'])
        acc         = float(metrics['accuracy'])
        data_size   = int(metrics['data_size'])

        filename = f'logs/{self.dataset}/clients_he_novo.csv' if self.homomorphic else f'logs/{self.dataset}/clients_novo.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'a') as file:
            file.write(f"{server_round}, {cid}, {loss}, {acc}, {train_time}, {data_size}, {delay}\n")


    def log_metrics_server(self, server_round, aggregation_time):
        filename = f'logs/{self.dataset}/server_he.csv' if self.homomorphic else f'logs/{self.dataset}/server.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'a') as file:
            file.write(f"{server_round}, {self.selection_time}, {aggregation_time}\n")


    def aggregate_fit(self, server_round, results, failures):		
        end_delay         = time.time()
        weights_results   = []
        agg_parameters    = 0
        parameters_list   = []
        total_examples    = 0 
        aggregation_start = time.time()

        if self.homomorphic:
            
            if self.packing: #packing aggregation
                aggregated_masks                   = []
                pack_mask_pair, total_examples     = get_pack_mask_pair(self, results)
                aggregated_packs, aggregated_masks = aggregate_packs_and_mask(pack_mask_pair, total_examples, self.onlysum)
                
                aggregated_packs    = serialize_packs(aggregated_packs)
                self.agg_parameters = pickle.dumps(aggregated_packs)
                self.agg_mask       = pickle.dumps(aggregated_masks)
                self.total_examples = total_examples
                
                return [], {}
                
            else: #no packing aggregation
                parameters_list, total_examples = get_cyphered_parameters(self, results)
                agg_parameters  = aggregated_cyphered_parameters(self, parameters_list, total_examples)

                if self.homomorphic_type == 'Paillier':
                    self.agg_parameters = pickle.dumps(agg_parameters)
                    self.total_examples = len(parameters_list)
                else:
                    self.agg_parameters = agg_parameters.serialize()
                    self.total_examples = total_examples
                #self.log_metrics_server(server_round, aggregation_time)
                return [], {}
        
        else: #plaintext aggregation
            total_examples = 0
            for _, fit_res in results:
                parameters_list.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                total_examples += int(fit_res.num_examples)
                #self.log_metrics_client(fit_res.metrics, server_round, end_delay)
                
            agg_parameters      = aggregate(parameters_list)
            aggregation_time    = time.time() - aggregation_start
            self.total_examples = total_examples
            #self.log_metrics_server(server_round, aggregation_time)

            return ndarrays_to_parameters(agg_parameters), {}
    
    def aggregate(self, results):
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Precompute the multiplicative inverse of num_examples_total
        inverse_num_examples_total = 1.0 / num_examples_total

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer using multiplication instead of division
        weights_prime = [
            reduce(np.add, layer_updates) * inverse_num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation losses using weighted average."""
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

        print(f"Round {server_round} aggregated loss: {loss_aggregated} aggregated accuracy: {sum(accuracies)/len(accuracies)}")

        return loss_aggregated, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {
            'he'    : self.agg_parameters,
            'round' : server_round,
            'mask'  : self.agg_mask,
            'total_examples' : str(self.total_examples),
        }  # {"server_round": server_round, "local_epochs": 1}

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

def main():
	
	server =  HEServer(num_clients     =  int(os.environ['NCLIENTS']), 
                       dirichlet_alpha =  float(os.environ['DIRICHLET_ALPHA']), 
                       dataset         =  os.environ['DATASET'], 
                       fraction_fit    =  float(os.environ['FRAC_FIT']),
                       homomorphic     = os.environ['HOMOMORPHIC'] == 'True',
                       packing         = os.environ['PACKING'] == 'True',
                       onlysum         = os.environ['ONLYSUM'] == 'True',
                       homomorphic_type = str(os.environ['HOMOMORPHIC_TYPE'])
            )

	fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=int(os.environ['NUM_ROUNDS'])),
        strategy=server, 
        )

if __name__ == '__main__':
	main()