import os
import re
from datetime import datetime

# Função para criar a pasta logs/data-horario
def create_log_directory():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = f"logs/{current_time}"
    os.makedirs(directory, exist_ok=True)
    return directory

# Função para extrair o valor da variável NOT_ENCRYPT_LAYERS como texto do arquivo client.py
def extract_not_encrypt_layers(filepath):
    with open("./client/client.py", 'r') as file:
        content = file.read()
    
    # Usando regex para capturar o valor da variável NOT_ENCRYPT_LAYERS
    match = re.search(r'self\.NOT_ENCRYPTED_LAYERS\s*=\s*(\d+)', content)
    if match:
        return match.group(1)
    else:
        return None

# Função para salvar o valor extraído em um arquivo txt
def save_to_log_file(directory, value):
    if value is not None:
        log_file_path = os.path.join(directory, "info.txt")
        with open(log_file_path, 'w') as file:
            file.write(f"NOT_ENCRYPTED_LAYERS = {value}")
        print(f"Valor salvo em {log_file_path}")
    else:
        print("Variável NOT_ENCRYPT_LAYERS não encontrada.")

# Caminho do arquivo client.py
client_file_path = "client.py"

# Criando o diretório de logs
log_directory = create_log_directory()

# Extraindo o valor de NOT_ENCRYPT_LAYERS
not_encrypt_value = extract_not_encrypt_layers(client_file_path)

# Salvando o valor em um arquivo txt
save_to_log_file(log_directory, not_encrypt_value)
