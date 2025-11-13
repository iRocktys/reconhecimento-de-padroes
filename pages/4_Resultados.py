# pages/4_Resultados.py
import streamlit as st

st.title("📈 Resultados Detalhados")

if st.session_state.last_results is not None:
    # --- Informações do Treinamento ---
    st.subheader(f"Modelo Treinado: {st.session_state.trained_model}")
    
    # Exibe a configuração do treinamento, se disponível
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
    st.dataframe(st.session_state.last_results, use_container_width=True)
    
    # --- Gráfico ---
    st.subheader(f"Gráfico de Evolução da Acurácia")
    st.line_chart(
        st.session_state.last_results,
        x="Epoch/Batch",
        y="Accuracy"
    )

else:
    st.info("Nenhum modelo foi treinado ainda. Prossiga para a aba **'Treinamento'** após o pré-processamento dos dados.")