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
from YPHE import YPHE_client
from FEDPHE import FEDPHE_client

def main():
    if os.environ["TYPE"] == "YPHE":
        client =  YPHE_client(
                cid             = int(os.environ['CID']), 
                niid            = bool(os.environ['NIID']), 
                dataset         = os.environ['DATASET'], 
                num_clients     = int(os.environ['NCLIENTS']), 
                dirichlet_alpha = float(os.environ['DIRICHLET_ALPHA'])
                )
    elif os.environ["TYPE"] == "FEDPHE":
         client =  FEDPHE_client(
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