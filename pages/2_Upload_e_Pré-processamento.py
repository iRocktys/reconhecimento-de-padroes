# pages/2_Upload_e_Pré-processamento.py
import streamlit as st
import pandas as pd
# Importa a função de pré-processamento do seu arquivo utils
from utils.preprocessing import perform_preprocessing 

st.title("📂 Upload e Pré-processamento do Dataset")
st.write("Envie seu arquivo CSV e prepare os dados para o treinamento.")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

# Lógica para carregar e processar o arquivo
if uploaded_file is not None:
    # 1. Carrega o DataFrame
    # Usa um dicionário para armazenar os dados e o nome do arquivo para verificação
    is_new_file = st.session_state.df_original is None or uploaded_file.name != st.session_state.df_original.get('name')
    
    if is_new_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_original = {'data': df.copy(), 'name': uploaded_file.name} # Salva cópia e nome
        st.session_state.df_processed = None # Reseta o processado
        st.success(f"Arquivo **{uploaded_file.name}** carregado! ({df.shape[0]} linhas)")
    else:
        df = st.session_state.df_original['data'].copy()
        st.info(f"Arquivo **{st.session_state.df_original['name']}** já está carregado.")
        
    st.subheader("Visualização dos Dados Originais")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Aplicar Pré-processamento")
    
    # 2. Botão para aplicar o pré-processamento
    if st.button("Aplicar Pré-processamento"):
        with st.spinner("Processando dados..."):
            # Chama a função de pré-processamento, passando os dados
            processed_df = perform_preprocessing(st.session_state.df_original['data'].copy())
            st.session_state.df_processed = processed_df
        
        st.success("Pré-processamento concluído!")
        
        st.subheader("Visualização dos Dados Processados")
        st.dataframe(st.session_state.df_processed.head(), use_container_width=True)
        st.info(f"O DataFrame processado tem {st.session_state.df_processed.shape[1]} colunas e {st.session_state.df_processed.shape[0]} amostras.")

# Lógica para exibir dados processados se já existirem
elif st.session_state.df_processed is not None:
    st.subheader("Dados Atualmente Processados")
    st.info(f"DataFrame carregado e processado pronto para a próxima etapa: **{st.session_state.df_processed.shape[0]} amostras**.")
    st.dataframe(st.session_state.df_processed.head(), use_container_width=True)