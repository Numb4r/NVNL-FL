import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_DIR="logs"
path_img="img"
TOTAL_USERS=2
from pathlib import Path
folders = [f for f in Path(LOG_DIR).iterdir() if f.is_dir()]

print(folders[0].parts[1])
def generate_client_train(cid,path):
    df = pd.read_csv(f"{path}/client_{cid}_train.csv")

    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(10, 6))
    
    ax1.plot(df.iloc[:,0], label='Acurácia', marker='o', color='b')
    ax1.set_title('Acurácia ao longo das Iterações')
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Acurácia')
    ax1.grid(True)
    ax1.legend()

    # Subplot para Loss
    ax2.plot(df.iloc[:,1], label='Loss', marker='x', color='r')
    ax2.set_title('Loss ao longo das Iterações')
    ax2.set_xlabel('Iterações')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()


    ax3.plot(df.iloc[:,2], label='Acurácia', marker='o', color='b')
    ax3.set_title('Acurácia ao longo das Iterações')
    ax3.set_xlabel('Iterações')
    ax3.set_ylabel('Tamanho vetor não criptografado')
    ax3.grid(True)
    ax3.legend()

    # Subplot para Loss
    ax4.plot(df.iloc[:,3], label='Loss', marker='x', color='r')
    ax4.set_title('Loss ao longo das Iterações')
    ax4.set_xlabel('Iterações')
    ax4.set_ylabel('Tamanho vetor criptografado')
    ax4.grid(True)
    ax4.legend()

    plt.savefig(f"{path_img}/{path.parts[1]}/client_{cid}_train.png")
    plt.close()
    
    
    

def generate_client_eval(cid,path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    df = pd.read_csv(f"{path}/client_{cid}_eval.csv")
    # Subplot para Acurácia
    ax1.plot(df.iloc[:,0], label='Acurácia', marker='o', color='b')
    ax1.set_title('Acurácia ao longo das Iterações')
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Acurácia')
    ax1.grid(True)
    ax1.legend()

    # Subplot para Loss
    ax2.plot(df.iloc[:,1], label='Loss', marker='x', color='r')
    ax2.set_title('Loss ao longo das Iterações')
    ax2.set_xlabel('Iterações')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    # Ajustar o layout para que os gráficos não se sobreponham
    plt.tight_layout()
    
    # Mostrar o gráfico
    plt.savefig(f"{path_img}/{path.parts[1]}/client_{cid}_eval.png")
    plt.close()
    # 

def generate_server_eval(path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    df = pd.read_csv(f"{path}/server_evaluate.csv")
    # Subplot para Acurácia
    ax1.plot(df.iloc[:,0], label='Acurácia', marker='o', color='b')
    ax1.set_title('Acurácia ao longo das Iterações')
    ax1.set_xlabel('Iterações')
    ax1.set_ylabel('Acurácia')
    ax1.grid(True)
    ax1.legend()

    # Subplot para Loss
    ax2.plot(df.iloc[:,1], label='Loss', marker='x', color='r')
    ax2.set_title('Loss ao longo das Iterações')
    ax2.set_xlabel('Iterações')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()

    # Ajustar o layout para que os gráficos não se sobreponham
    plt.tight_layout()
    
    # Mostrar o gráfico
    plt.savefig(f"{path_img}/{path.parts[1]}/server_eval.png")
    plt.close()
    

if not os.path.exists('img'):
    os.makedirs('img')
for f in folders:
    csv_files = list(Path(f).glob("*.csv"))
    
    # Se não houver arquivos .csv, imprime um erro
    if not csv_files:
        print(f"Erro: Nenhum arquivo .csv encontrado na pasta {f}")
        continue  # Pule para a próxima pasta
    if not os.path.exists(f'img/{f.parts[1]}'):
        os.makedirs(f"img/{f.parts[1]}")
    generate_server_eval(f)
    for c in range(TOTAL_USERS):
        generate_client_eval(c,f)
        generate_client_train(c,f)



# generate_client_eval(0)
# generate_client_train(0)

# generate_client_eval(1)
# generate_client_train(1)
# generate_server_eval()
#     # Adicionando título e rótulos
#     plt.title('Gráfico de Linha: Valor ao longo do tempo')
#     plt.xlabel('Tempo')
#     plt.ylabel('Valor')

#     # Exibindo legenda
#     plt.legend()

#     # Mostrando o gráfico
#     
# # Lendo o arquivo CSV
# df = pd.read_csv('.csv')

# # Exibindo as primeiras linhas do DataFrame
# print(df.head())



# Gerando um gráfico de linha com duas colunas
# Suponha que o CSV tenha uma coluna 'tempo' e outra 'valor'
