import os
from optparse import OptionParser
import random


def add_server_info(clients, rounds, solution, dataset, frac_fit, alpha):
    server_str = f"  server:\n\
    image: 'flhe:server'\n\
    container_name: fl_server\n\
    environment:\n\
      - SERVER_IP=0.0.0.0:9999\n\
      - NCLIENTS={clients}\n\
      - NUM_ROUNDS={rounds}\n\
      - SOLUTION={solution}\n\
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

def add_client_info(cid, nclients, solution, dataset, niid, alpha,
                    start2share,percentage,technique):
    client_str = f"  client-{cid}:\n\
    image: 'flhe:client'\n\
    environment:\n\
      - SERVER_IP=fl_server:9999\n\
      - CID={cid}\n\
      - SOLUTION={solution}\n\
      - DATASET={dataset}\n\
      - NIID={niid}\n\
      - NCLIENTS={nclients}\n\
      - DIRICHLET_ALPHA={alpha}\n\
      - START2SHARE={start2share}\n\
      - PERCENTAGE={percentage}\n\
      - TECHNIQUE={technique}\n\
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

def add_shared_volume():
    shared_volume = "volumes:\n\
      vol_solution:\n\
        driver: local\n\
        driver_opts:\n\
          type: nfs\n\
          o: addr=10.10.10.113,nolock,soft,rw\n\
          device: :/opt/nodes/FLHE\n\
    \n\n"

    return shared_volume

def add_prometheus_grafana_cadvisor():
    prometheus_str = "  cadvisor:\n\
    image: google/cadvisor:latest\n\
    container_name: cadvisor\n\
    ports:\n\
      - 8080:8080\n\
    volumes:\n\
      - /:/rootfs:ro\n\
      - /var/run:/var/run:ro\n\
      - /sys:/sys:ro\n\
      - /var/lib/docker/:/var/lib/docker:ro\n\
    restart: unless-stopped\n\
    \n\
  prometheus:\n\
    image: prom/prometheus:latest\n\
    container_name: prometheus\n\
    volumes:\n\
      - ./prometheus.yml:/etc/prometheus/prometheus.yml\n\
    ports:\n\
      - 9090:9090\n\
    restart: unless-stopped\n\
    \n\
  grafana:\n\
    image: grafana/grafana:latest\n\
    container_name: grafana\n\
    ports:\n\
      - 3000:3000\n\
    volumes:\n\
      - grafana-storage:/var/lib/grafana\n\
    environment:\n\
      - GF_SECURITY_ADMIN_PASSWORD=admin # Defina a senha do Grafana\n\
    restart: unless-stopped\n\
    \n\
    "

    return prometheus_str


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
    parser.add_option("",   "--start2share",        dest="start2share",   default=1)
    parser.add_option("",   "--he",                 dest="he",   default=False)
    parser.add_option("",   "--he-type",            dest="he_type",   default='Full')  
    parser.add_option("",   "--technique",          dest="technique",  default='topk')  
    parser.add_option("",   "--percentage",            dest="percentage",   default=0.3)    

    (opt, args) = parser.parse_args()
    
    with open(f'composes/{opt.solution}-{opt.dataset}-d={opt.dirichilet}-t={opt.technique}-p={opt.percentage}.yaml', 'w') as dockercompose_file:
        header = f"version: '3'\nservices:\n\n"

        dockercompose_file.write(header)
                                    
        server_str = add_server_info(opt.clients, opt.rounds, opt.solution, opt.dataset, opt.frac_fit, opt.dirichilet)

        dockercompose_file.write(server_str)

        for cid in range(int(opt.clients)):
            client_str = add_client_info(cid, opt.clients, opt.solution, opt.dataset, opt.niid, opt.dirichilet, 
                                         opt.start2share,opt.percentage,opt.technique)    

            dockercompose_file.write(client_str)
            
        # prometheus_str = add_prometheus_grafana_cadvisor()
        # dockercompose_file.write(prometheus_str)

if __name__ == '__main__':
	main()