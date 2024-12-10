import os
import re
import yaml
from datetime import datetime

# Função para criar a pasta logs/data-horario
def create_log_directory():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f"logs/{current_time}"
    os.makedirs(directory, exist_ok=True)
    return directory

# Função para extrair o valor da variável NOT_ENCRYPT_LAYERS como texto do arquivo client.py
def type_technique(data):
    
    # technique=''
    
    if 'services' in data and 'server' in data['services']:
        server_type = None
        # Verifica as variáveis de ambiente para o servidor
        for env_var in data['services']['server'].get('environment', []):
            if env_var.startswith('TYPE='):
                server_type = env_var.split('=')[1]
                return f'TYPE={server_type}\n'
        #         break
        # if server_type:
        #     types['server'] = server_type
    # Obtém os tipos dos clientes
    # if 'services' in data:
    #     client_types = {}
    #     for service_name, service_data in data['services'].items():
    #         if service_name.startswith('client'):
    #             client_type = None
    #             # Verifica as variáveis de ambiente para o cliente
    #             for env_var in service_data.get('environment', []):
    #                 if env_var.startswith('TYPE='):
    #                     client_type = env_var.split('=')[1]
    #                     break
    #             if client_type:
    #                 client_types[service_name] = client_type
    #     if client_types:
    #         types['clients'] = client_types
        # string_return = f"server = {types['server']}\n"
        # string_return+="".join([f"{item[0]} = {item[1]}\n" for item in types['clients'].items()])
        
        # return string_return


def fedphe_block_number():
    with open("./teste-mnist-2.yaml",'r') as file:
        data = yaml.safe_load(file)
    if 'services' in data and 'server' in data['services']:
        for env_var in data['services']['server'].get('environment', []):
            if env_var.startswith('TYPE='):
                server_type = env_var.split('=')[1]
                if server_type != 'FEDPHE':
                    
                    return ''
                break
    with open("./client/FEDPHE.py", 'r') as file:
        content = file.read()
    
    
    match = re.search(r'self\.QNT_BLOCOS\s*=\s*(\d+)', content)
    print(match)
    if match:
        return f'QNT_BLOCOS = {match.group(1)}\n'
    else:
        return None
# Função para salvar o valor extraído em um arquivo txt
def save_to_log_file(directory, list_value):
    log_file_path = os.path.join(directory, "info.txt")
    with open(log_file_path, 'w') as file:
        for value in list_value:
            if value is not None:
                    file.write(f"{value}")
            print(f"Valor salvo em {log_file_path}")


    
         
        #         break
        # if server_type:
        #     types['server'] = server_type
def extract_info_yaml():
    with open("./teste-mnist.yaml",'r') as file:
            data = yaml.safe_load(file)
    string_r = ''
    infos_server = ['NCLIENTS',"NUM_ROUNDS","TYPE","DATASET","DIRICHLET_ALPHA"]
    info_client = ['NIID','DIRICHLET_ALPHA']
    if 'services' in data and 'server' in data['services']:
        # Verifica as variáveis de ambiente para o servidor
        for info in infos_server:
            for env_var in data['services']['server'].get('environment', []):
                
                
                if env_var.startswith(f'{info}='):
                    str_extract = env_var.split('=')[1]
                    string_r+=f'{info}={str_extract}\n'
    else:
        string_r+=f'Não foi possivel extrair as informações sobre o servidor\n'
    if 'services' in data and 'client-0' in data['services']:
        for info in info_client:
            for env_var in data['services']['client-0'].get('environment',[]):
                 if env_var.startswith(f'{info}='):
                    str_extract = env_var.split('=')[1]
                    string_r+=f'{info}={str_extract}\n'
    else:
        string_r+=f'Não foi possivel extrair as informações sobre os clientes\n'
    return string_r
# Caminho do arquivo client.py
client_file_path = "client.py"

# Criando o diretório de logs
log_directory = create_log_directory()





# Salvando o valor em um arquivo txt
save_to_log_file(log_directory, [
    extract_info_yaml(),
    
    fedphe_block_number(),

    ])
