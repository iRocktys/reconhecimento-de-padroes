# pages/3_Pré-processamento.py
import streamlit as st
import pandas as pd
from utils.preprocessing import perform_preprocessing 

st.title("⚙️ Pré-processamento do Dataset")
st.write("Aplique o pré-processamento necessário aos dados carregados.")

if st.session_state.df_original is None:
    st.warning("⚠️ Por favor, primeiro selecione e carregue um dataset na aba **'Base de Dados'**.")
    st.stop()

st.info(f"Dataset carregado para pré-processamento: **{st.session_state.selected_csv_name}** com {st.session_state.df_original['data'].shape[0] if isinstance(st.session_state.df_original, dict) else st.session_state.df_original.shape[0]} linhas.")
st.subheader("Visualização dos Dados Originais")
# MUDANÇA AQUI: use_container_width=True -> width='stretch'
st.dataframe(st.session_state.df_original.copy().head(), width='stretch') 

st.markdown("---")

if st.button("Aplicar Pré-processamento"):
    with st.spinner("Processando dados..."):
        # Garante que passamos o DataFrame, seja ele armazenado como dict ou diretamente
        df_to_process = st.session_state.df_original['data'].copy() if isinstance(st.session_state.df_original, dict) else st.session_state.df_original.copy()
        processed_df = perform_preprocessing(df_to_process)
        st.session_state.df_processed = processed_df
    
    st.success("Pré-processamento concluído!")
    
    st.subheader("Visualização dos Dados Processados")
    # MUDANÇA AQUI: use_container_width=True -> width='stretch'
    st.dataframe(st.session_state.df_processed.head(), width='stretch') 
    st.info(f"O DataFrame processado tem {st.session_state.df_processed.shape[1]} colunas e {st.session_state.df_processed.shape[0]} amostras.")

elif st.session_state.df_processed is not None:
    st.subheader("Dados Atualmente Processados")
    st.info(f"DataFrame processado pronto para a próxima etapa: **{st.session_state.df_processed.shape[0]} amostras**.")
    # MUDANÇA AQUI: use_container_width=True -> width='stretch'
    st.dataframe(st.session_state.df_processed.head(), width='stretch')