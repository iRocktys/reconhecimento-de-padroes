import streamlit as st
from utils.style import load_custom_css
load_custom_css("style.css")

# --- INICIALIZAÇÃO CRÍTICA DO ESTADO ---
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
# Chaves usadas em páginas subsequentes (Pré-processamento)
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
# ----------------------------------------
st.set_page_config(
    page_title="App IDS Stream Mining", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Conteúdo da Página Inicial Simplificado ---

st.title("🛡️ Sistema de Detecção de Intrusão (IDS) com Stream Mining")
st.markdown("""
Bem-vindo à plataforma de modelagem de **dados de fluxo (stream)** para detecção de intrusão.

### ⚙️ Como Começar:

Utilize o menu lateral (sidebar) para navegar entre os passos do fluxo de trabalho:

1.  **Base de Dados:** Selecione um dataset CSV pré-carregado e visualize seus dados.
2.  **Pré-processamento:** Prepare o dataset para o treinamento.
3.  **Treinamento:** Configure e treine modelos de Stream Mining.
4.  **Resultados Detalhados:** Visualize a evolução da acurácia e o desempenho do modelo.

Clique em **'Base de Dados'** na barra lateral para iniciar.
""")

if st.session_state.df_original is not None:
    st.info(f"Dataset selecionado: **{st.session_state.selected_csv_name}** com {st.session_state.df_original.shape[0]} amostras.")
elif st.session_state.df_processed is not None: # Caso o df_original tenha sido carregado e depois a página inicial seja acessada
    st.info(f"Dataset processado pronto para treinamento: {st.session_state.df_processed.shape[0]} amostras.")