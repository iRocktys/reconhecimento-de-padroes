# Stream-IDS: Experimentação Visual em Stream Mining para Detecção de Intrusão

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![CapyMOA](https://img.shields.io/badge/Lib-CapyMOA-green)

Este repositório contém uma aplicação visual desenvolvida em **Streamlit** para a experimentação, análise e comparação de algoritmos de **Aprendizado de Máquina Online (Stream Mining)** aplicados a Sistemas de Detecção de Intrusão (IDS).

A ferramenta utiliza a biblioteca **CapyMOA** para integrar algoritmos e oferece um pipeline completo: desde o pré-processamento do dataset CICDDoS2019 até a avaliação prequencial com detecção visual de *concept drift*.

---

## 🛠️ Pré-requisitos

Antes de começar, certifique-se de ter instalado em sua máquina:

1.  **Python 3.10 ou superior**: [Download Python](https://www.python.org/downloads/)
2.  **Java JDK (Obrigatório)**: O CapyMOA depende da JVM para executar os algoritmos do MOA. Certifique-se de ter o Java instalado e configurado no PATH do sistema.
    * *Verifique no terminal:* `java -version`

---

## 🚀 Instalação

Siga os passos abaixo para configurar o ambiente:

### 1. Clone o repositório
```bash
git clone [https://github.com/iRocktys/reconhecimento-de-padroes.git](https://github.com/iRocktys/reconhecimento-de-padroes.git)
cd reconhecimento-de-padroes
```

### 2. Crie um Ambiente Virtual (Recomendado)
Para evitar conflitos de bibliotecas, crie e ative um ambiente virtual:

* **Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

* **Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências
Utilize o arquivo `requirements.txt` para instalar todas as bibliotecas necessárias automaticamente:

```bash
pip install -r requirements.txt
```

---

## ▶️ Como Executar

Com o ambiente virtual ativado e as dependências instaladas, inicie a aplicação Streamlit. Como o projeto é dividido em páginas, recomenda-se iniciar pelo módulo de **Base de Dados**:

```bash
python -m streamlit run 1_Home.py
```

> **Nota:** O navegador abrirá automaticamente no endereço `http://localhost:8501`.

---

## 📂 Estrutura do Projeto

* `2_Base_de_Dados.py`: Módulo de ingestão, downsampling dinâmico e seleção de arquivos.
* `3_Pré-processamento.py`: Pipeline de limpeza, imputação e seleção de features via correlação.
* `4_Modelos.py`: Configuração dos hiperparâmetros dos algoritmos e geração de streams sintéticos.
* `5_Avaliação.py`: Dashboard de execução prequencial, métricas acumulativas e visualização de *drifts*.
* `utils/`: Diretório contendo funções auxiliares de processamento, carregamento e estilização.
