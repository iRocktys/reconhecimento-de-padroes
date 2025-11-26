import streamlit as st
from utils.style import load_custom_css
load_custom_css("style.css")

if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = "N/A"
if 'selected_csv_name' not in st.session_state:
    st.session_state.selected_csv_name = "N/A"
if 'target_col' not in st.session_state:
    st.session_state.target_col = 'label'
if 'timestamp_col' not in st.session_state:
    st.session_state.timestamp_col = 'timestamp'
if 'cols_to_remove' not in st.session_state:
    st.session_state.cols_to_remove = []
if 'features_to_keep' not in st.session_state:
    st.session_state.features_to_keep = []
if 'imputation_method' not in st.session_state:
    st.session_state.imputation_method = 'Mediana'
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_filepath' not in st.session_state:
    st.session_state.processed_filepath = None
if 'processed_amostras' not in st.session_state:
    st.session_state.processed_amostras = 0
if 'file_to_analyze' not in st.session_state:
    st.session_state.file_to_analyze = None 

st.set_page_config(
    page_title="IDS Stream Mining", 
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Sistema de Detecção de Intrusão com Stream Mining")
st.header("Sobre o Dataset e a Metodologia", divider="rainbow")
st.subheader("O Dataset: CICDDoS2019")
st.markdown("""
Esta aplicação utiliza o **CICDDoS2019**, um dos datasets mais modernos e abrangentes para a detecção de ataques de Negação de Serviço (DDoS). Ele foi criado pelo *Canadian Institute for Cybersecurity (CIC)* e é amplamente utilizado pela comunidade acadêmica e de segurança.

**Como funciona:** O dataset é composto por capturas de tráfego de rede (arquivos PCAP) que foram processadas e transformadas em fluxos. Cada linha nos arquivos `.csv` representa um fluxo de rede (uma "conversa" entre dois computadores) e é descrita por mais de 80 *features* (características), como duração do fluxo, número de pacotes, tamanho dos pacotes, etc.

**Onde encontrar:** O dataset completo está disponível publicamente em várias fontes, incluindo a página oficial da universidade e o Kaggle:
* **Página Oficial (UNB):** [https://www.unb.ca/cic/datasets/ddos-2019.html](https://www.unb.ca/cic/datasets/ddos-2019.html)
* **Versão no Kaggle (CSV):** [https://www.kaggle.com/datasets/rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files](https://www.kaggle.com/datasets/rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files)
""")
st.subheader("A Metodologia: Machine Learning e Stream Mining")
st.markdown("""
O volume de dados de rede é gigantesco e contínuo. Por isso, uma abordagem de *Machine Learning* tradicional (onde treinamos o modelo uma única vez com todos os dados) não é ideal.

Neste projeto, exploramos a metodologia de **Stream Mining** (Mineração de Dados de Fluxo). O objetivo é construir um modelo que possa ser treinado e fazer previsões em tempo real, analisando cada fluxo de rede **individualmente, à medida que ele chega**.

As próximas páginas deste aplicativo o guiarão pelo processo de carregar, processar, treinar e avaliar um modelo de *Stream Mining* com esses dados.
""")
st.header("Sobre o Autor e este Projeto", divider="rainbow")
st.markdown(f"""
Este aplicativo está sendo desenvolvido pelo autor Leandro M. Tosta como projeto prático 
para a disciplina de **Reconhecimento de Padrões** do programa de Mestrado 
em Ciência da Computação da **Universidade Estadual de Londrina (UEL)**.

**Orientador:** Prof. Dr. Bruno Zarpelão.

O código-fonte completo e a documentação deste projeto estão disponíveis publicamente no GitHub.

[https://github.com/iRocktys/reconhecimento-de-padroes](httpsS://github.com/iRocktys/reconhecimento-de-padroes)
""")