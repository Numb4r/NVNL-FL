configure:
	docker build -t fl_server server/.
	docker build -t fl_client client/.
run:
	python3.10 generate_log_setup.py && docker compose -f teste-mnist-2.yaml up
graph:
	python3.10 generate_graphs.py 
keys:
	python3.10 generate_keys.py
cleanlogs:
	rm -rf logs/