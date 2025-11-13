import streamlit as st

# --- Configuração da Página ---
st.set_page_config(
    page_title="App IDS Stream Mining", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inicialização do st.session_state (MANTIDA) ---
# --- Inicialização do st.session_state ---
# Garante que todas as chaves existam antes de serem acessadas em outras páginas.
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = "N/A"
if 'hyperparameters' not in st.session_state:
    st.session_state.hyperparameters = {'confidence': 0.01, 'grace_period': 200} 
if 'epochs' not in st.session_state:
    st.session_state.epochs = 10
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Hoeffding Tree"
if 'training_config' not in st.session_state:
    st.session_state.training_config = {}

# LINHA DE CORREÇÃO CRÍTICA: Inicializa a chave ausente
if 'selected_csv_name' not in st.session_state: 
    st.session_state.selected_csv_name = None


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