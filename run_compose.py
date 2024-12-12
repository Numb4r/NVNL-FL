import os
import sys

SOLS = ['BFV', 'CKKS', 'BatchCrypt', 'FedPHE', 'plaintext']	

COMPOSES = [
    # 'BFV-MNIST-10.yaml',
    # 'CKKS-MNIST-10.yaml',
    # 'FedPHE-MNIST-10.yaml',
    # 'plaintext-MNIST-10.yaml',
    'BatchCrypt-MNIST-10.yaml',
]

def create_compose():
    for sol in SOLS:
        print(f'Creating compose for {sol}')
        cmd = f'python create_dockercompose.py -c 10 -s {sol} -d MNIST -r 100 -f 0.5 --niid True --dirichilet 0.1'
        os.system(cmd)            
            
def run_compose():
    for file in COMPOSES:
        cmd = f'docker compose -f {file} up'
        os.system(cmd)

def main():
    # create_compose()
    run_compose()
    
main()