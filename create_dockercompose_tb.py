import os
from optparse import OptionParser
import random


def add_server_info(clients, rounds, solution, dataset, frac_fit, alpha, he):
    server_str = f"  server:\n\
    image: 'flhe:server'\n\
    container_name: fl_server\n\
    environment:\n\
      - SERVER_IP=0.0.0.0:9999\n\
      - NCLIENTS={clients}\n\
      - NUM_ROUNDS={rounds}\n\
      - SOLUTION_NAME={solution}\n\
      - DATASET={dataset}\n\
      - FRAC_FIT={frac_fit}\n\
      - DIRICHLET_ALPHA={alpha}\n\
      - HOMOMORPHIC={he}\n\
    volumes:\n\
      - /opt/nodes/FLHE/logs:/logs:rw\n\
      - /opt/nodes/FLHE/server:/server:rw\n\
      - /opt/nodes/FLHE/context:/context:r\n\
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
                    start2share, homomorphic, homomorphic_type):
    client_str = f"  client-{cid}:\n\
    image: 'teste_tenseal:latest'\n\
    environment:\n\
      - SERVER_IP=fl_server:9999\n\
      - CID={cid}\n\
      - SOLUTION_NAME={solution}\n\
      - DATASET={dataset}\n\
      - NIID={niid}\n\
      - NCLIENTS={nclients}\n\
      - DIRICHLET_ALPHA={alpha}\n\
      - START2SHARE={start2share}\n\
      - HOMOMORPHIC={homomorphic}\n\
      - HOMOMORPHIC_TYPE={homomorphic_type}\n\
    volumes:\n\
      - vol_logs:/TenSEAL/logs:rw\n\
      - vol_client:/TenSEAL/client:r\n\
      - vol_context:/context:r\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==worker\n\
    command: python client/client.py\n\
          \n\n"

    return client_str

def add_shared_volumes():
    shared_volumes = "volumes:\n\
  vol_logs:\n\
    driver: local\n\
    driver_opts:\n\
      type: nfs\n\
      o: addr=10.10.10.113,nolock,soft,rw\n\
      device: :/opt/nodes/FLHE/logs\n\
      \n\
  vol_client:\n\
    driver: local\n\
    driver_opts:\n\
      type: nfs\n\
      o: addr=10.10.10.113,nolock,soft,rw\n\
      device: :/opt/nodes/FLHE/client\n\
      \n\
  vol_server:\n\
    driver: local\n\
    driver_opts:\n\
      type: nfs\n\
      o: addr=10.10.10.113,nolock,soft,rw\n\
      device: :/opt/nodes/FLHE/server\n\
      \n\
  vol_context:\n\
    driver: local\n\
    driver_opts:\n\
      type: nfs\n\
      o: addr=10.10.10.113,nolock,soft,rw\n\
      device: :/opt/nodes/FLHE/context\n\
  \n\n"

    return shared_volumes

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

    (opt, args) = parser.parse_args()

    with open(f'{opt.solution}-{opt.dataset}-{opt.clients}.yaml', 'a') as dockercompose_file:
        header = f"version: '3'\nservices:\n\n"

        dockercompose_file.write(header)
                                    
        server_str = add_server_info(opt.clients, opt.rounds, opt.solution, opt.dataset, opt.frac_fit, opt.dirichilet, opt.he)

        dockercompose_file.write(server_str)

        for cid in range(int(opt.clients)):
            client_str = add_client_info(cid, opt.clients, opt.solution, opt.dataset, opt.niid, opt.dirichilet, 
                                         opt.start2share, opt.he, opt.he_type)    

            dockercompose_file.write(client_str)
        
        volumes = add_shared_volumes()
        dockercompose_file.write(volumes) 
        # prometheus_str = add_prometheus_grafana_cadvisor()
        # dockercompose_file.write(prometheus_str)

if __name__ == '__main__':
	main()