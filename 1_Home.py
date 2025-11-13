import streamlit as st
import streamlit_shadcn_ui as st_ui # Para botões interativos
import time # Adicionado para simular interatividade, se necessário

# --- Configuração da Página ---
st.set_page_config(
    page_title="App IDS Stream Mining", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    # Valores padrão para Hoeffding Tree
    st.session_state.hyperparameters = {'confidence': 0.01, 'grace_period': 200} 
if 'epochs' not in st.session_state:
    st.session_state.epochs = 10
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Hoeffding Tree"


# --- Conteúdo da Página Inicial ---

st.title("🛡️ Sistema de Detecção de Intrusão com Stream Mining")
st.markdown("""
Bem-vindo à plataforma para análise e modelagem de **dados de fluxo (stream)** para detecção de intrusão.
Utilize os passos abaixo para carregar seu dataset, treinar modelos de *machine learning* incremental e visualizar os resultados de acurácia em tempo real.
""")

st.markdown("---")
