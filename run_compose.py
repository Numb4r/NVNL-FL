import os
import sys

SOLS = ['BFV', 'CKKS', 'BatchCrypt', 'FedPHE', 'plaintext']	

# COMPOSES = [
#     # 'BFV-MNIST-10.yaml',
#     # 'CKKS-MNIST-10.yaml',
#     # 'FedPHE-MNIST-10.yaml',
#     # 'plaintext-MNIST-10.yaml',
#     'BatchCrypt-MNIST-10.yaml',
# ]

def create_compose():  
    
    os.makedirs("./composes",exist_ok=True)
    for sol in SOLS:
        for dirichilet in [0.1,1]:
            for technique in ["topk","robin_round","slided_window","weight_random"]:
                for percentage in [0.3,0.5,0.8]:
                    for dataset in ["MNIST","FASHION_MNIST"]:
                        print(f'Creating compose for {sol}')
                        cmd = f'python3.10 create_dockercompose.py -c 10 -s {sol} -d {dataset} -r 100 -f 0.5 --niid True --dirichilet {dirichilet} --percentage {percentage} --technique {technique}'
                        os.system(cmd)                    
def run_compose(COMPOSES = []):
    diretorio = "./composes/"
    if COMPOSES == []:
        COMPOSES =     arquivos = [arquivo for arquivo in os.listdir(diretorio) if os.path.isfile(os.path.join(diretorio, arquivo))]
    
    # print(COMPOSES)
    for file in COMPOSES:
        cmd = f"mv {diretorio+file} ./{file}"
        os.system(cmd)
        cmd = f'echo "{file}" >> last.txt'
        os.system(cmd)
        cmd = f'docker compose -f {file} up'
        os.system(cmd)
        cmd = f"mv ./{file} {diretorio+file}"
        os.system(cmd)


    print(cmd)
    #     cmd = f'docker compose -f {file} up'
    #     os.system(cmd)
def verify():
    with  open('last.txt','r') as file:
        linhas = file.readlines()
    linhas = [linha.strip() for linha in linhas]
    diretorio = "./composes"
    COMPOSES =     arquivos = [arquivo for arquivo in os.listdir(diretorio) if os.path.isfile(os.path.join(diretorio, arquivo))]
    lset = set(linhas)
    print(len(lset),len(COMPOSES))
    if linhas == COMPOSES:
        print("Terminado")
    else:
        diff1 = [item for item in linhas if item not in COMPOSES]
        # Elementos presentes em COMPOSES, mas não em linhas
        diff2 = [item for item in COMPOSES if item not in linhas]

        print("Os arrays são diferentes.")
        
        print(f"Elementos em linhas e não em COMPOSES: {diff1}")
        
        print(f"Elementos em COMPOSES e não em linhas: {diff2}")
# FedPHE-MNIST-d=0.1-t=robin_round-p=0.8.yaml
import pandas as pd

def verificar_csvs_em_subdiretorio(diretorio_base):
    arquivos_nao_carregados = []
    arquivos_round_100 = {}
    
    # Percorrer todos os arquivos em subdiretórios
    for root, _, files in os.walk(diretorio_base):
        for file in files:
            if file.endswith('.csv'):
                caminho_arquivo = os.path.join(root, file)
                try:
                    df = pd.read_csv(caminho_arquivo)
                    
                    # Verificar se a coluna 'round' existe
                    if 'round' in df.columns:
                        ultimo_valor = df['round'].iloc[-1]
                        arquivos_round_100[file] = ultimo_valor == 100
                    else:
                        arquivos_nao_carregados.append(file)
                except Exception as e:
                    arquivos_nao_carregados.append(file)
    
    # Contar quantos arquivos têm o último valor igual a 100
    arquivos_com_100 = [k for k, v in arquivos_round_100.items() if v]
    arquivos_sem_100 = [k for k, v in arquivos_round_100.items() if not v]
    
    # print(f"Arquivos com último valor 'round' igual a 100: {arquivos_com_100}")
    print(f"Arquivos com último valor 'round' diferente de 100: {arquivos_sem_100}")
    print(f"Arquivos que não puderam ser carregados: {arquivos_nao_carregados}")








def extrair_parametros_yaml(nome_arquivo):
    """Extrai parâmetros do nome do arquivo YAML."""
    partes = nome_arquivo.replace(".yaml", "").split("-")
    if len(partes) < 5:
        return None  # Nome de arquivo inválido
    
    metodo = partes[0]
    dataset = partes[1]
    alpha = partes[2].split("=")[1]
    tecnica = partes[3].split("=")[1]
    porcentagem = partes[4].split("=")[1]
    
    return {
        "metodo": metodo,
        "dataset": dataset,
        "alpha": alpha,
        "tecnica": tecnica,
        "porcentagem": porcentagem,
        "nome_arquivo": nome_arquivo
    }

def extrair_parametros_csv(nome_arquivo):
    """Extrai parâmetros do nome do arquivo CSV."""
    partes = nome_arquivo.replace(".csv", "").split(" - ")
    
    if len(partes) < 3:
        return None  # Nome de arquivo inválido
    
    
    
    if partes[0].startswith("train_"):
        tecnica_p_d = partes[0][len("train_"):]
    else:
        tecnica_p_d = partes[0][len("evaluate_"):]
    
    tecnica = tecnica_p_d.split("-")[0]
    porcentagem = tecnica_p_d.split("-")[1]
    dataset = partes[1]
    metodo = partes[2]
    alpha = tecnica_p_d.split("-")[2]
    
    return {
        "metodo": metodo,
        "dataset": dataset,
        "tecnica": tecnica,
        "alpha":alpha,
        "porcentagem": porcentagem
    }

def verificar_yaml_faltantes(diretorio_yaml, diretorio_csv):
    # Ler todos os arquivos YAML
    yaml_files = [f for f in os.listdir(diretorio_yaml) if f.endswith('.yaml')]
    yaml_parametros = [extrair_parametros_yaml(f) for f in yaml_files if extrair_parametros_yaml(f)]
    
    # Ler todos os arquivos CSV
    csv_files = []
    for root, _, files in os.walk(diretorio_csv):
        csv_files.extend([f for f in files if f.endswith('.csv')])
    
    csv_parametros = [extrair_parametros_csv(f) for f in csv_files if extrair_parametros_csv(f)]
    
    # Comparar YAMLs com CSVs
    yaml_faltantes = []
    for yaml_param in yaml_parametros:
        encontrado = any(
            yaml_param["metodo"].upper() == csv_param["metodo"].upper() and
            yaml_param["dataset"] == csv_param["dataset"] and
            yaml_param["tecnica"] == csv_param["tecnica"] and
            yaml_param["porcentagem"] == csv_param["porcentagem"] and
            yaml_param["alpha"] == csv_param["alpha"]
            for csv_param in csv_parametros
        )
        if not encontrado:
            yaml_faltantes.append(yaml_param["nome_arquivo"])
    
    return yaml_faltantes

diretorio_yaml = "composes"
diretorio_csv = "logs/"
# a = extrair_parametros_yaml("BatchCrypt-FASHION_MNIST-d=0.1-t=topk-p=0.3.yaml")
# b= extrair_parametros_csv("evaluate_topk-0.3-0.1 - FASHION_MNIST - batchcrypt.csv")

# print(a["metodo"].upper() == b["metodo"].upper() )
# print(a["dataset"] == b["dataset"] )
# print(a["tecnica"] == b["tecnica"] )
# print(a["porcentagem"] == b["porcentagem"]  )
# print(a["alpha"] == b["alpha"])
        



def main():
    create_compose()
    run_compose()
    # verify()
    # verificar_csvs_em_subdiretorio("logs")
    faltantes = verificar_yaml_faltantes(diretorio_yaml, diretorio_csv)
    print("YAMLs que faltam rodar:")
    lista = []
    for f in faltantes:
        lista.append(f)
    print(len(lista))
    run_compose(lista)
main()
