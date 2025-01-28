PYTHON := python3.10

configure:
	docker build -t flhe:server server/.
	docker build -t flhe:client client/.

run:
	docker compose -f FedPHE-FASHIONMNIST-10.yaml up

graph:
	$(PYTHON) generate_graphs.py 

keys:
	$(PYTHON) generate_keys.py

cleanlogs:
	rm -rf logs/