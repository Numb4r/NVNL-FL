import docker
import time
import os

# Inicializando o cliente Docker
client = docker.from_env()

# Nome do arquivo de log
log_file = "monitoramento_ram_docker.log"

# Função para escrever dados no arquivo de log
def write_log(container_name, memory_usage):
    with open(log_file, 'a') as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {container_name} - Memória usada: {memory_usage} MB\n")

# Função para monitorar o uso de RAM de um container específico
def monitor_container(container):
    try:
        stats = container.stats(stream=False)
        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # Convertendo bytes para MB
        print(memory_usage)
        write_log(container.name, memory_usage)
    except Exception as e:
        print(f"Erro ao monitorar container {container.name}: {str(e)}")

# Função principal de monitoramento
def monitor_docker_containers():
    # Limpando o arquivo de log existente
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Monitorar os containers a cada 10 segundos
    while True:
        # Buscando containers com a imagem fl_server
        fl_server_container = client.containers.list(filters={"ancestor": "fl_server"})
        
        # Buscando containers com a imagem fl_client
        fl_client_containers = client.containers.list(filters={"ancestor": "fl_client"})
        
        # Monitorando o container fl_server
        if fl_server_container:
            monitor_container(fl_server_container[0])  # Assumindo que há apenas um container fl_server
        
        # Monitorando os containers fl_client
        for container in fl_client_containers:
            monitor_container(container)

        # Pausa de 10 segundos entre as leituras
        time.sleep(2)

if __name__ == "__main__":
    monitor_docker_containers()
