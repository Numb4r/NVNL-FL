FROM python:3.10-slim
#FROM arm64v8/python:3.11.2 

RUN pip install -U pip

#ADD TenSEAL /TenSEAL
# ADD tenseal-0.3.15-cp311-cp311-linux_aarch64.whl ./tenseal-0.3.15-cp311-cp311-linux_aarch64.whl
# RUN pip install tenseal-0.3.15-cp311-cp311-linux_aarch64.whl
# Update package lists and install git
#RUN apt-get update && apt-get install -y cmake

#RUN git clone --recursive https://github.com/OpenMined/TenSEAL.git
#WORKDIR /TenSEAL
#RUN pip install .
#RUN pip install "pandas<2.0.0" numpy  tensorflow flwr=="1.4.0"  flwr-datasets flwr_datasets[vision] matplotlib gmpy2

RUN pip install "pandas<2.0.0" numpy  tensorflow flwr=="1.4.0"  tenseal flwr-datasets flwr_datasets[vision] matplotlib gmpy2

CMD ["python", "-u", "client/client.py"]
