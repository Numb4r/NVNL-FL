FROM python:3.10-slim
#FROM arm64v8/python:3.9

RUN pip install -U pip
RUN pip install tenseal pandas numpy  tensorflow flwr=="1.4.0" flwr-datasets flwr_datasets[vision] matplotlib seaborn

CMD ["python", "-u", "server/server.py"]


#docker build -t allanmsouza/flhe:client .