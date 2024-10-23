import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
A MEDIÇÃO DO TAMANHO DO MODELO ESTA ERRADA
EU CONSIDEREI APENAS O TAMANHO DA CIFRAGEM, MAS NO CASO DO FEDAVG EU PEGO O VETOR INTEIRO
SOMAR AS OUTRAS CAMADAS DA CIFRAGEM COM AS CAMADAS SEM CIFRAR




MEDIR O TEMPO DE TREINAMENTO 
 
"""

def accuracy_graph():
    fhe_sever = pd.read_csv('FHE/server_evaluate.csv')
    phe_sever = pd.read_csv('PHE/server_evaluate.csv')
    fedavg_server = pd.read_csv('FedAvg/server_evaluate.csv')
    fig, ax = plt.subplots(1,1,figsize=(10, 6))
    ax.plot(fhe_sever.iloc[:,0], label='FHE', marker='o', color='b')
    ax.plot(phe_sever.iloc[:,0], label='PHE', marker='o', color='r')
    ax.plot(fedavg_server.iloc[:,0], label='FedAvg', marker='o', color='g')
    ax.set_xlabel("Rodadas de agregação")
    ax.set_ylabel("Acurácia")
    ax.grid(True)
    ax.legend()
    plt.title('Acurácia do modelo agregado por solução')
    plt.savefig("comparacao_acuracia.png")
    plt.close()

# accuracy_graph()
def tempo_cifragem():
    fhe_client = pd.read_csv('FHE/client_0_train.csv')
    phe_client = pd.read_csv('PHE/client_0_train.csv')
    fedavg_client = pd.read_csv('FedAvg/client_0_train.csv')


    solucoes = ['FedAvg', 'FHE', 'PHE']
    tempo_cifragem = [  np.mean(fedavg_client.iloc[:,colunacifragem]),
                        np.mean(fhe_client.iloc[:,colunacifragem]),
                        np.mean(phe_client.iloc[:,colunacifragem])]
      # Valores estimados
    tempo_decifragem =  [  np.mean(fedavg_client.iloc[:,colunadecifragem]),
                        np.mean(fhe_client.iloc[:,colunadecifragem]),
                        np.mean(phe_client.iloc[:,colunadecifragem])]

    tempo_treinamento =  [  np.mean(fedavg_client.iloc[:,colunatreinamento]),
                            np.mean(fhe_client.iloc[:,colunatreinamento]),
                            np.mean(phe_client.iloc[:,colunatreinamento])]
    
    # Configurando a  barra empilhada horizontal
    ind = np.arange(len(solucoes))  # Posições das barras
    largura = 0.35  # Largura das barras

    # Plotando as barras horizontais
    plt.barh(ind, tempo_cifragem, label='Tempo Cifragem', color='orange')
    plt.barh(ind, tempo_decifragem, left=tempo_cifragem, label='Tempo Decifragem', color='gray')
    plt.barh(ind, tempo_treinamento, left=np.array(tempo_cifragem) + np.array(tempo_decifragem), 
            label='Tempo Treinamento', color='blue')

    # Adicionando detalhes
    plt.xlabel('Tempo (s)')
    plt.title('Tempo de Execução por Solução')
    plt.yticks(ind, solucoes)
    plt.legend()

    # Exibindo o gráfico
    plt.show()


def tamanho_modelo():
    fhe_client = pd.read_csv('FHE/client_0_train.csv')
    phe_client = pd.read_csv('PHE/client_0_train.csv')
    fedavg_client = pd.read_csv('FedAvg/client_0_train.csv')


    solucoes = ['FedAvg', 'FHE', 'PHE']


    fedavg_tamanho = np.mean(fedavg_client.iloc[:,tamanho])
    fhe_tamanho = np.mean(fhe_client.iloc[:,tamanho])
    phe_tamanho = np.mean(phe_client.iloc[:,tamanho])
    
    # Configurando a  barra empilhada horizontal
    ind = np.arange(len(solucoes))  # Posições das barras
    largura = 0.35  # Largura das barras

    # Plotando as barras horizontais
    plt.barh(ind, fedavg_tamanho, color='orange')
    plt.barh(ind, tempo_decifragem, color='gray')
    plt.barh(ind, tempo_treinamento, color='blue')

    # Adicionando detalhes
    plt.xlabel('Tamanho em bytes')
    plt.title('Tamanho do modelo em memoria por Solução')
    plt.yticks(ind, solucoes)
    plt.legend()

    # Exibindo o gráfico
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Dados (valores estimados com base na imagem)
solucoes = ['FedAvg', 'FHE Lastasd Layer', 'FHE']
tempo_cifragem = [0.1, 0.5, 1.5]  # Valores estimados
tempo_decifragem = [0.0, 0.3, 1.0]
tempo_treinamento = [0.05, 0.2, 0.8]

# Configurando a barra empilhada
ind = np.arange(len(solucoes))  # Posições das barras
largura = 0.35  # Largura das barras

# Plotando as barras
plt.bar(ind, tempo_cifragem, largura, color='orange')
plt.bar(ind, tempo_decifragem, largura, bottom=tempo_cifragem, label='Tempo Decifragem', color='gray')
# plt.bar(ind, tempo_treinamento, largura, bottom=np.array(tempo_cifragem) + np.array(tempo_decifragem), 
        # label='Tempo Treinamento', color='blue')

# Adicionando detalhes
plt.ylabel('Tempo (s)')
plt.title('Tempo de Execução por Solução')
plt.xticks(ind, solucoes)
plt.legend()

# Exibindo o gráfico
plt.show()
