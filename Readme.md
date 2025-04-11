# NVNL-FL: Aprendizado Federado Eficiente com Criptografia Homomórfica

Este repositório contém o código do **NVNL-FL**, uma solução eficiente para Aprendizado Federado (FL) em cenários cross-silo utilizando Criptografia Homomórfica (HE). A proposta combina técnicas de empacotamento, esparsificação e janela deslizante para reduzir overheads computacionais e de comunicação, garantindo privacidade dos dados sem comprometer o desempenho do modelo.

## 🚀 Funcionalidades

- **Aprendizado Federado Seguro**: Utiliza Criptografia Homomórfica (HE) para preservar a privacidade dos dados.
- **Redução de Overheads**: Implementa técnicas de empacotamento e esparsificação para otimizar a comunicação e o processamento.
- **Robustez em Dados Não-IID**: Desempenho garantido mesmo em cenários com dados heterogêneos.

---

## 📋 Pré-requisitos

- **Docker** e **Docker Compose**: Para configurar o ambiente em contêineres.
- **Python 3.8+** com as bibliotecas:
  - `tensorflow`
  - `flower`
  - `tenseal`
  - `numpy`
  - `matplotlib`
  - `gmpy2`

---

## ⚙️ Utilização

### 1. Clone o repositório
```bash
$ git clone https://github.com/SeuUsuario/nvnl-fl.git
$ cd nvnl-fl
```

### 2. Gere as imagens Docker
Use o Docker configurar o ambiente do servidor e dos clientes.
```bash
$ docker build -t flhe:server server/.
$ docker build -t flhe:client client/.

```

### 3. Gere o arquivo Yaml do ambiente 
Se preferir rodar localmente (fora do Docker), instale as dependências manualmente:
```bash
$ python3 create_dockercompose.py -c 10 -s FedPHE -d MNIST -r 100 -f 0.5 --niid True --dirichilet 0.1 --percentage 0.3 --technique robin_round
# -c : Numero de clientes
# -s : Solução [BFV, CKKS, BatchCrypt, FedPHE, plaintext]
# -d : Dataset [Mnist,FashionMnist]
# -r : Numero de rodadas de comunicação
# -f : Fator de clientes agregados [0,...,1]
# --niid : Cenario IID ou Não-IID  [True,False]
# --dirichilet: Fator Dirichilet [0,...,1]
# --percentage: Porcentagem de pacotes enviados para agregação [0,...,1]
# --technique: Tecnica para seleção de pacotes [topk,robin_round,slided_window,weight_random]
```

---
## 4. Gere as chaves dos esquema de criptografia
```bash 
$ python3 generate_keys.py
``` 
## 5. Execução
```bash
$ docker compose -f nome_do_arquivo.yaml up
```
## Reproduza nossos resultados:
Nesse repositorio, é possivel encontrar todos os componente necessarios para reproduzir a pesquisa. A instrução abaixo reproduz resultados Não-IID, com alpha Dirichilet 10% e 30% de agregação dos pacotes usando a janela deslizante com passo de mesmo tamanho da janela (Robin Round)
```bash
$ python3.10 create_dockercompose.py -c 10 -s FedPHE -d MNIST -r 100 -f 0.5 --niid True --dirichilet 0.1 --percentage 0.3 --technique robin_round
```
Para reproduzir os outros resultados, modifique:
1. `--dirichilet` para `1`
2. `--percentage` para `0.5` e `0.8`
3. `--technique` pode ser trocada para `slided_window`, que utiliza passo 50% dos pacotes agregados.
---
## 📂 Estrutura do Repositório

```
├── cliente/
│   ├── client.py        # Código base dos clientes FL
│   ├── literature.py    # Implementação de técnicas da literatura
│   ├── models.py        # Implementação dos modelos ML
│   ├── client_utils.py  # Funções auxiliares do cliente
│   ├── encryption/
│   │   ├── bfv.py           # Implementação da tecnica BFV
│   │   ├── ckks.py          # Implementação da tecnica CKKS
│   │   └── paillier.py      # Implementação da tecnica Paillier
├── server/
│   ├── server.py        # Código base dos server FL
│   ├── server_utils.py  # Funções auxiliares do cliente
├── logs/ # Resultados das execuções
├── docker-compose.yml   # Configuração Docker
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação do projeto
```

---


## 📊 Resultados Experimentais
Os experimentos utilizaram os datasets MNIST e FashionMNIST, treinados com uma DNN de 128, 64 e 32 neurônios para o MNIST e a arquitetura LeNet-5 para o FashionMNIST. O treinamento foi realizado com Stochastic Gradient Descent (SGD) e função de perda Sparse Categorical Crossentropy. Para criptografia homomórfica, foram testados os esquemas CKKS, BFV, Batchcrypt,FedPHE e NVNL. O número total de clientes foi 10, com 50% selecionados por rodada ao longo de 100 rodadas de comunicação. A heterogeneidade dos dados foi controlada pela distribuição de Dirichlet, variando entre IID (α = 1) e não-IID (α = 0,1)

| **Dados**  | **Soluções**  | **Acurácia (%)** | **TX Bytes (Mb)** | **Tempo (s)** | **Acurácia (%)** | **TX Bytes (Mb)** | **Tempo (s)** |
|------------|--------------|------------------|------------------|--------------|------------------|------------------|--------------|
|            |              | **MNIST**        |                  |              | **FashionMNIST** |                  |              |
| **IID**    | FedAvg       | 0.97             | 449.49           | 0.59         | 0.90             | 272.25           | 2.06         |
|            | BFV          | 0.97             | 3009.08          | 1.21         | 0.90             | 1719.48          | 2.46         |
|            | CKKS         | 0.97             | 4651.85          | 0.89         | 0.90             | 2658.22          | 2.22         |
|            | BatchCrypt   | 0.97             | 571.77           | 32.14        | 0.88             | 317.68           | 20.35        |
|            | FedPHE       | 0.94             | 996.94           | 0.72         | 0.81             | 664.63           | 2.26         |
|            | NVNL-FL      | 0.96             | 996.95           | 0.76         | 0.85             | 664.63           | 2.36         |
| **Não-IID**| FedAvg       | 0.97             | 449.49           | 0.53         | 0.84             | 272.25           | 1.69         |
|            | BFV          | 0.95             | 3009.08          | 1.15         | 0.79             | 1719.48          | 2.21         |
|            | CKKS         | 0.95             | 4651.85          | 0.82         | 0.81             | 2658.22          | 1.94         |
|            | BatchCrypt   | 0.95             | 571.77           | 31.96        | 0.83             | 317.68           | 19.99        |
|            | FedPHE       | 0.49             | 996.94           | 0.71         | 0.35             | 664.63           | 2.26         |
|            | NVNL-FL      | 0.92             | 996.95           | 0.77         | 0.75             | 664.63           | 2.36         |

### Dados IID
![Dados IID](/img/mnist_iid.png)
### Dados Não IID
![Dados não IID](/img/mnist_niid.png)
---



## 📜 Referências

Este trabalho é baseado no artigo:

**"O Que Não é Visto, Não é Lembrado: Aprendizado Federado Eficiente com Criptografia Homomórfica"**  
Yuri Dimitre D. Faria, Allan M. de Souza  
Instituto de Computação - Universidade Estadual de Campinas

---

## 🤝  Agradecimentos

Expresso minha gratidão aos seguintes projetos e seus contribuintes pelos seus trabalhos e contribuições acadêmicas:


- [FedPHE](https://github.com/lunan0320/FedPHE) - pelo artigo "Efficient and Straggler-Resistant Homomorphic Encryption for Heterogeneous Federated Learning", que foi uma base essencial para esse projeto.
- [BatchCrypt](https://github.com/marcoszh/BatchCrypt) -  pelo artigo "BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning".
- [FLASHE](https://github.com/SamuelGong/FLASHE) - pela sua contribuição tecnicas de otimização de esparcificação de pacotes.


## 💬 Citação
```bibtex
@inproceedings{nvnl_fl,
 author = {Yuri D. Faria and Luiz F. Bittencourt and Leandro A. Villas and Allan M. de Souza},
 title = {O Que Não é Visto, Não é Lembrado: Aprendizado Federado Eficiente com Criptografia Homomórfica},
 booktitle = {Anais do XLIII Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos},
 location = {Natal/RN},
 year = {2025},
 keywords = {},
 issn = {2177-9384},
 pages = {1--14},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
}
```




---

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
