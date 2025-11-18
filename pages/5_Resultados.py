# pages/5_Resultados.py
import streamlit as st
from utils.style import load_custom_css
load_custom_css("style.css")

st.title("📈 Resultados Detalhados")

if st.session_state.last_results is not None:
    # --- Informações do Treinamento ---
    st.subheader(f"Modelo Treinado: {st.session_state.trained_model}")
    
    if st.session_state.get('training_config'):
        config = st.session_state.training_config
        st.markdown(f"""
        * **Modelo Selecionado:** `{config.get('model', 'N/A')}`
        * **Número de Batches (Simulação):** `{config.get('epochs', 'N/A')}`
        * **Hiperparâmetros:** `{config.get('hyperparameters', 'N/A')}`
        """)
        st.markdown("---")
        
    # --- Tabela de Resultados ---
    st.subheader("Resultados de Acurácia por Batch")
    # MUDANÇA AQUI: use_container_width=True -> width='stretch'
    st.dataframe(st.session_state.last_results, width='stretch')
    
    # --- Gráfico ---
    st.subheader(f"Gráfico de Evolução da Acurácia")
    # MUDANÇA AQUI: use_container_width=True -> width='stretch'
    st.line_chart(
        st.session_state.last_results,
        x="Epoch/Batch",
        y="Accuracy",
        width='stretch' 
    )

else:
    st.info("Nenhum modelo foi treinado ainda. Prossiga para a aba **'Treinamento'** após o pré-processamento dos dados.")