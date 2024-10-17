import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Função para ler e processar o arquivo
def processar_arquivo(caminho_arquivo):
    # Criar listas para armazenar os dados
    timestamps = []
    containers = []
    memoria_usada = []
    
    # Ler o arquivo linha por linha
    with open(caminho_arquivo, 'r') as file:
        for linha in file:
            # Exemplo de linha: "2024-10-17 14:35:32 - fl_server - Memória usada: 138.328125 MB"
            partes = linha.split(" - ")
            timestamp = partes[0]
            container = partes[1]
            memoria = float(partes[2].split(":")[1].strip().split()[0])
            
            # Armazenar os dados
            timestamps.append(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
            containers.append(container)
            memoria_usada.append(memoria)
    
    # Criar um DataFrame com os dados
    df = pd.DataFrame({
        'timestamp': timestamps,
        'container': containers,
        'memoria_usada_MB': memoria_usada
    })
    
    # Normalizar o tempo (primeiro timestamp = 0, último = tempo total corrido)
    tempo_inicial = df['timestamp'].min()
    df['tempo_normalizado'] = (df['timestamp'] - tempo_inicial).dt.total_seconds()
    
    return df

# Função para gerar o gráfico
def gerar_grafico(df):
    plt.figure(figsize=(10, 6))
    
    # Iterar sobre cada container único e plotar os dados
    for container in df['container'].unique():
        df_container = df[df['container'] == container]
        plt.plot(df_container['tempo_normalizado'], df_container['memoria_usada_MB'], label=container)
    
    # Adicionar títulos e rótulos
    plt.title('Uso de Memória dos Containers ao Longo do Tempo (Tempo Normalizado)')
    plt.xlabel('Tempo Corrido (segundos)')
    plt.ylabel('Memória usada (MB)')
    plt.legend()
    plt.tight_layout()
    
    # Mostrar o gráfico
    plt.savefig(f"{caminho_arquivo}.png")
    plt.show()

# Caminho do arquivo
caminho_arquivo = 'monitoramento_ram_docker-partial.log'

# Processar o arquivo e gerar o gráfico
df = processar_arquivo(caminho_arquivo)
gerar_grafico(df)
