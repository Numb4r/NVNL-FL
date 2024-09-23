from client.client import HEClient
from server.server import HEServer
import flwr as fl
import numpy as np


NCLIENTS        = 2
DATASET         = 'mnist'
NIID            = False
DIRICHLET_ALPHA = 0.5
FRACTION_FIT    = 1.0

def create_client(cid):
        client = HEClient(cid, NIID, DATASET, NCLIENTS, DIRICHLET_ALPHA)
        return client.to_client()

class HESimulation():
    def __init__(self):
        self.server  = HEServer(num_clients=NCLIENTS, dirichlet_alpha=DIRICHLET_ALPHA, dataset=DATASET, fraction_fit=FRACTION_FIT)
        
    def run_simulation(self):
        ray_args = {
			"include_dashboard"   : False,
			"ignore_reinit_error" : True,
            "num_gpus"            : 0,
            "num_cpus"            : 12,
            # 'object_store_memory' : 2136287232.0, 
            # 'memory'              : 4272574464.0,
		}

        fl.simulation.start_simulation(
            client_fn     = create_client,
            num_clients   = NCLIENTS,
            config        = fl.server.ServerConfig(num_rounds=10),
            strategy      = self.server,
            ray_init_args = ray_args)

HESimulation().run_simulation()