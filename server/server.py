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


class HEServer(fl.server.strategy.FedAvg):
    def __init__(self, num_clients, dirichlet_alpha, dataset, fraction_fit=1.0):
        self.num_clients     = num_clients
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset         = dataset
        self.context         = self.get_server_context()
        self.agg_parameters  = []
 
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
        if len(self.agg_parameters) > 0:
            data2send = self.agg_parameters
        
        config = {
            'he': data2send,
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
        
        for _, fit_res in results:
            client_id      = str(fit_res.metrics['cid'])
            parameters     = ts.ckks_tensor_from(self.context, fit_res.metrics['he']) 
            parameters_list.append((parameters, int(fit_res.num_examples)))
            total_examples  += int(fit_res.num_examples)
            
        for parameters, num_examples in parameters_list:
            weights         = num_examples / total_examples
            agg_parameters  = agg_parameters + (parameters * weights)

        
        self.agg_parameters = agg_parameters.serialize()

        return [], {}
    
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
            'he': self.agg_parameters,
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
                       fraction_fit    =  float(os.environ['FRAC_FIT'])
            )

	fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=int(os.environ['NUM_ROUNDS'])),
        strategy=server, 
        )

if __name__ == '__main__':
	main()