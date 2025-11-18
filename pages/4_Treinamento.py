# pages/4_Treinamento.py
import streamlit as st
import streamlit_shadcn_ui as st_ui
import pandas as pd
import numpy as np
import time
from utils.models import HoeffdingTreeModel
from utils.style import load_custom_css
load_custom_css("style.css")

st.title("🔬 Treinamento de Modelos de Stream Mining")

if st.session_state.df_processed is None:
    st.warning("⚠️ Por favor, primeiro processe um dataset na aba **'Pré-processamento'**.")
    st.stop()
    
st.success(f"Dados prontos para treinamento: **{st.session_state.df_processed.shape[0]}** amostras.")

# --- Layout Centralizado (sem alteração de largura aqui) ---
col1, col2, col3 = st.columns([1, 2, 1]) 

with col2:
    with st_ui.card(title="Configurar Hiperparâmetros", 
                    description="Defina a configuração do treinamento e do modelo.", 
                    class_name="w-full"):
        
        # ... (Configuração de Hiperparâmetros) ...

        current_model = st.session_state.selected_model
        current_conf = st.session_state.hyperparameters.get('confidence', 0.01)
        current_grace = st.session_state.hyperparameters.get('grace_period', 200)
        current_epochs = st.session_state.epochs
        
        model_options = ["Hoeffding Tree", "Adaptive Random Forest", "SVM (P-SMO)"]
        selected_model = st_ui.select("Modelo", options=model_options, default_value=current_model)
        
        if selected_model == "Hoeffding Tree":
            confidence_value = st_ui.slider("Hoeffding Bound Confidence", min_value=0.001, max_value=0.5, step=0.001, 
                                            default_value=[current_conf])[0]
            grace_period_str = st_ui.input("Grace Period (Amostras)", type="number", 
                                          default_value=str(current_grace))
            
            try:
                grace_period = int(grace_period_str)
            except ValueError:
                st.error("O Grace Period deve ser um número inteiro válido.")
                st.stop()
                
            hyperparameters = {'confidence': confidence_value, 'grace_period': grace_period}
        else:
             hyperparameters = {'default': 'default'}

        epochs_str = st_ui.input("Número de Épocas/Batches (Simulação)", type="number", 
                                default_value=str(current_epochs))
        
        try:
            epochs = int(epochs_str)
        except ValueError:
            st.error("O Número de Épocas/Batches deve ser um número inteiro válido.")
            st.stop()
            
        st.session_state.selected_model = selected_model
        st.session_state.hyperparameters = hyperparameters
        st.session_state.epochs = epochs

        train_button = st_ui.button("Iniciar Treinamento", class_name="w-full mt-4")

    # LÓGICA DE TREINAMENTO
    if train_button:
        if epochs <= 0:
            st.error("O número de Épocas/Batches deve ser maior que zero.")
            st.stop()
            
        if selected_model == "Hoeffding Tree":
            model = HoeffdingTreeModel(st.session_state.hyperparameters) 
        else:
            st.warning(f"Modelo '{selected_model}' não implementado. Usando simulação básica (Hoeffding Tree).")
            model = HoeffdingTreeModel(st.session_state.hyperparameters)
            
        with st.spinner(f"Treinando {model.name} em {epochs} batches..."):
            time.sleep(1) 
            
            accuracy_array = model.train_on_stream(st.session_state.df_processed, epochs)
            
            results_df = pd.DataFrame({
                "Epoch/Batch": np.arange(1, len(accuracy_array) + 1),
                "Accuracy": accuracy_array
            })
            
            st.session_state.last_results = results_df
            st.session_state.trained_model = model.name 
            st.session_state.training_config = { 
                'model': st.session_state.selected_model,
                'hyperparameters': st.session_state.hyperparameters,
                'epochs': st.session_state.epochs
            }
        
        st.success(f"Treinamento do **{model.name}** concluído!")
        st.balloons()

    # Exibe o gráfico se houver resultados
    if st.session_state.last_results is not None:
        st.subheader(f"Evolução da Acurácia - {st.session_state.trained_model}")
        
        st.line_chart(
            st.session_state.last_results,
            x="Epoch/Batch",
            y="Accuracy",
            width='stretch' 
        )