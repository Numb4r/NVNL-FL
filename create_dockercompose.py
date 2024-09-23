import os
from optparse import OptionParser
import random


def add_server_info(clients, rounds, solution, dataset, frac_fit, alpha):
    server_str = f"  server:\n\
    image: 'allanmsouza/flhe:server'\n\
    container_name: fl_server\n\
    environment:\n\
      - SERVER_IP=0.0.0.0:9999\n\
      - NCLIENTS={clients}\n\
      - NUM_ROUNDS={rounds}\n\
      - SOLUTION_NAME={solution}\n\
      - DATASET={dataset}\n\
      - FRAC_FIT={frac_fit}\n\
      - DIRICHLET_ALPHA={alpha}\n\
    volumes:\n\
      - ./server:/server:r\n\
      - ./context:/context:r\n\
      - ./logs:/logs\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==manager\n\
    \n\n"

    return server_str

def add_client_info(cid, nclients, solution, dataset, niid, alpha):
    client_str = f"  client-{cid}:\n\
    image: 'allanmsouza/flhe:client'\n\
    environment:\n\
      - SERVER_IP=fl_server:9999\n\
      - CID={cid}\n\
      - SOLUTION_NAME={solution}\n\
      - DATASET={dataset}\n\
      - NIID={niid}\n\
      - NCLIENTS={nclients}\n\
      - DIRICHLET_ALPHA={alpha}\n\
    volumes:\n\
      - ./client:/client:r\n\
      - ./context:/context:r\n\
      - ./logs:/logs\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==worker\n\
          \n\n"

    return client_str

def main():

    parser = OptionParser()
    # python create_dockercompose.py -c 2 -s teste -d mnist -r 10 --dirichilet 0.5 --frac-fit 1 --niid False
    parser.add_option("-c", "--clients",            dest="clients",  default=0)
    parser.add_option("-s", "--sol",                dest="solution", default=0)
    parser.add_option("-d", "--dataset",            dest="dataset",  default='MNIST')
    parser.add_option("-r", "--rounds",             dest="rounds",   default=100)
    parser.add_option("",   "--dirichilet",         dest="dirichilet",   default=1)
    parser.add_option("-f", "--frac-fit",           dest="frac_fit",   default=1)
    parser.add_option("",   "--niid",               dest="niid",   default=False)
    

    (opt, args) = parser.parse_args()

    with open(f'{opt.solution}-{opt.dataset}-{opt.clients}.yaml', 'a') as dockercompose_file:
        header = f"version: '3'\nservices:\n\n"

        dockercompose_file.write(header)
                                    
        server_str = add_server_info(opt.clients, opt.rounds, opt.solution, opt.dataset, opt.frac_fit, opt.dirichilet)

        dockercompose_file.write(server_str)

        for cid in range(int(opt.clients)):
            client_str = add_client_info(cid, opt.clients, opt.solution, opt.dataset, opt.niid, opt.dirichilet)    

            dockercompose_file.write(client_str)

if __name__ == '__main__':
	main()