PYTHON := python3.10

configure:
	docker build -t fl_server server/.
	docker build -t fl_client client/.

run:
	$(PYTHON) generate_log_setup.py && docker compose -f teste-mnist-2.yaml up

graph:
	$(PYTHON) generate_graphs.py 

keys:
	$(PYTHON) generate_keys.py

cleanlogs:
	rm -rf logs/