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
from YPHE import YPHE_server
from FEDPHE import FEDPHE_server

def main():
    if os.environ["TYPE"] == "YPHE":
        server =  YPHE_server(num_clients     =  int(os.environ['NCLIENTS']), 
                        dirichlet_alpha =  float(os.environ['DIRICHLET_ALPHA']), 
                        dataset         =  os.environ['DATASET'], 
                        fraction_fit    =  float(os.environ['FRAC_FIT'])
            )
    elif os.environ["TYPE"] == "FEDPHE":
         
         server =  FEDPHE_server(num_clients     =  int(os.environ['NCLIENTS']), 
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