import streamlit as st
import pandas as pd
import altair as alt 
import os
from utils.style import load_custom_css
from utils.training import get_models # Apenas importa o construtor

# --- Configuração da Página ---
st.set_page_config(
    page_title="Configuração de Treinamento", 
    page_icon="🏋️",
    layout="centered" 
)
load_custom_css("style.css")

# --- Renderização da Página ---
st.title("🏋️ 4. Configuração do Treinamento")

# --- Passo 1: Verificar se o Stream existe ---
if 'stream_data' not in st.session_state or st.session_state.stream_data is None:
    st.error("Nenhum *stream* de dados encontrado.")
    st.warning("Por favor, vá para a página '2. Pré-processamento' e execute o pipeline para criar um stream antes de continuar.")
    st.stop()

# --- CORREÇÃO DO ERRO AQUI ---
# Acessa o DataFrame 'X' final salvo na etapa anterior
if 'X_final_df' in st.session_state and st.session_state.X_final_df is not None:
    total_instances = st.session_state.X_final_df.shape[0]
else:
    # Fallback caso algo dê errado no processamento
    st.error("Erro ao carregar dados processados (X_final_df). Retorne ao Passo 2.")
    st.stop()
# --- FIM DA CORREÇÃO ---
    
stream = st.session_state.stream_data
st.success(f"Stream de dados carregado com sucesso! ({total_instances:,} instâncias)")
st.markdown("---")


# --- Passo 2: Configurações Globais de Avaliação (Request 2) ---
st.header("1. Parâmetros Globais de Avaliação", divider="rainbow")
st.markdown("Defina os parâmetros que serão usados na próxima etapa ('Avaliação') para rodar o(s) modelo(s) no stream.")

global_params = {}
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        global_params["MAX_INSTANCES"] = st.number_input(
            "Máximo de Instâncias (MAX_INSTANCES)",
            min_value=1000,
            value=min(10000, total_instances), # Padrão: 10k ou o total
            max_value=total_instances,
            step=1000,
            help="O número total de instâncias do stream que serão usadas na avaliação."
        )
    with c2:
        global_params["WINDOW_SIZE"] = st.number_input(
            "Janela de Avaliação (WINDOW_SIZE)",
            min_value=100,
            value=500,
            step=100,
            help="A janela deslizante (em instâncias) para calcular a acurácia."
        )
    
    c3, c4 = st.columns(2)
    with c3:
        # Renomeado de DELAY_LENGTH para 0 para bater com seu exemplo
        delay_value = st.number_input(
            "Atraso de Rótulo (DELAY_LENGTH)",
            min_value=0,
            value=0, # Padrão 0 (sem atraso)
            step=10,
            help="Simula um atraso (em instâncias) para o rótulo (y) chegar. 0 = sem atraso."
        )
        # Converte 0 para None, como no seu exemplo
        global_params["DELAY_LENGTH"] = delay_value if delay_value > 0 else None
        
    with c4:
        global_params["LABEL_PROBABILITY"] = st.slider(
            "Probabilidade de Rótulo (LABEL_PROBABILITY)",
            min_value=0.0,
            max_value=1.0,
            value=1.0, # Padrão 1.0 (todos os rótulos)
            step=0.05,
            help="Simula a chegada parcial de rótulos (ex: 0.5 = 50% dos rótulos chegam). 1.0 = todos os rótulos."
        )

# --- Passo 3: Seleção e Configuração dos Modelos ---
st.header("2. Seleção e Configuração dos Modelos", divider="rainbow")
st.markdown("Escolha um ou mais modelos de *Stream Mining* para configurar. Os parâmetros definidos aqui serão usados na próxima etapa de 'Avaliação'.")

all_model_names = [
    "LeveragingBagging", 
    "HoeffdingAdaptiveTree", 
    "AdaptiveRandomForest",
    "HoeffdingTree"
]

selected_models = st.multiselect(
    "Selecione os modelos para configurar:",
    options=all_model_names,
    default=all_model_names[0] 
)

hyperparams = {}

if not selected_models:
    st.warning("Por favor, selecione pelo menos um modelo para configurar.")
else:
    tabs = st.tabs(selected_models)
    
    for i, model_name in enumerate(selected_models):
        with tabs[i]:
            st.subheader(f"Hiperparâmetros: {model_name}", anchor=False)
            hyperparams[model_name] = {}
            
            if model_name == "LeveragingBagging":
                c1, c2 = st.columns(2)
                hyperparams[model_name]["ensemble_size"] = c1.number_input("Tamanho do Ensemble (ensemble_size)", min_value=1, value=100, step=10, key=f"{model_name}_ens")
                hyperparams[model_name]["random_seed"] = c2.number_input("Random Seed", value=1, step=1, key=f"{model_name}_rs")
            
            elif model_name == "HoeffdingAdaptiveTree":
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["grace_period"] = c1.number_input("Período de Graça (grace_period)", min_value=1, value=200, step=10, key=f"{model_name}_gp")
                hyperparams[model_name]["confidence"] = c2.number_input("Confiança (confidence)", min_value=0.0, max_value=1.0, value=0.01, step=0.01, format="%.2f", key=f"{model_name}_conf")
                hyperparams[model_name]["tie_threshold"] = c3.number_input("Limiar de Empate (tie_threshold)", min_value=0.0, max_value=1.0, value=0.05, step=0.05, format="%.2f", key=f"{model_name}_tie")
                hyperparams[model_name]["leaf_prediction"] = st.selectbox("Preditor da Folha (leaf_prediction)", ['NaiveBayes', 'NaiveBayesAdaptive', 'MC'], index=1, key=f"{model_name}_leaf")
                hyperparams[model_name]["nb_threshold"] = st.number_input("Limiar NaiveBayes (nb_threshold)", value=0, key=f"{model_name}_nb")

            elif model_name == "AdaptiveRandomForest":
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["ensemble_size"] = c1.number_input("Tamanho do Ensemble (ensemble_size)", min_value=1, value=100, step=10, key=f"{model_name}_ens")
                hyperparams[model_name]["max_features"] = c2.number_input("Max Features (max_features)", min_value=0.1, max_value=1.0, value=0.6, step=0.1, key=f"{model_name}_maxf")
                hyperparams[model_name]["lambda_param"] = c3.number_input("Lambda (lambda_param)", min_value=1.0, value=6.0, step=0.5, key=f"{model_name}_lambda")
                hyperparams[model_name]["disable_drift_detection"] = st.checkbox("Desabilitar Drift Interno", value=False, key=f"{model_name}_drift")

            elif model_name == "HoeffdingTree":
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["grace_period"] = c1.number_input("Período de Graça (grace_period)", min_value=1, value=200, step=10, key=f"{model_name}_gp")
                hyperparams[model_name]["confidence"] = c2.number_input("Confiança (confidence)", min_value=0.0, max_value=1.0, value=0.01, step=0.01, format="%.2f", key=f"{model_name}_conf")
                hyperparams[model_name]["tie_threshold"] = c3.number_input("Limiar de Empate (tie_threshold)", min_value=0.0, max_value=1.0, value=0.05, step=0.05, format="%.2f", key=f"{model_name}_tie")
                hyperparams[model_name]["leaf_prediction"] = st.selectbox("Preditor da Folha (leaf_prediction)", ['NaiveBayes', 'MC'], index=0, key=f"{model_name}_leaf")
                hyperparams[model_name]["nb_threshold"] = st.number_input("Limiar NaiveBayes (nb_threshold)", value=0, key=f"{model_name}_nb")

            # Configurações de Drift (comuns a todos)
            with st.expander("Configurar Detectores de Drift (DDM, ADWIN, ABCD)"):
                st.subheader("DDM (Drift Detection Method)", anchor=False)
                c1_ddm, c2_ddm, c3_ddm = st.columns(3)
                hyperparams[model_name]["ddm_min_instances"] = c1_ddm.number_input("Min. Instâncias", min_value=10, value=30, step=1, key=f"{model_name}_ddm_min")
                hyperparams[model_name]["ddm_warning_level"] = c2_ddm.number_input("Nível de Aviso", min_value=0.0, value=2.0, step=0.1, key=f"{model_name}_ddm_warn")
                hyperparams[model_name]["ddm_out_control_level"] = c3_ddm.number_input("Nível de Controle", min_value=0.0, value=3.0, step=0.1, key=f"{model_name}_ddm_out")

                st.subheader("ADWIN (Adaptive Windowing)", anchor=False)
                hyperparams[model_name]["adwin_delta"] = st.number_input("Delta", min_value=0.0, value=0.002, step=0.001, format="%.3f", key=f"{model_name}_adwin_delta", help="Delta de confiança. Menor = mais sensível.")
                
                st.subheader("ABCD (Drift Detector)", anchor=False)
                c1_abcd, c2_abcd = st.columns(2)
                hyperparams[model_name]["abcd_delta_drift"] = c1_abcd.number_input("Delta Drift", min_value=0.0, value=0.002, step=0.001, format="%.3f", key=f"{model_name}_abcd_drift")
                hyperparams[model_name]["abcd_delta_warn"] = c2_abcd.number_input("Delta Aviso", min_value=0.0, value=0.01, step=0.01, format="%.2f", key=f"{model_name}_abcd_warn")

# --- Passo 4: Salvar Configurações (Request 3) ---
st.header("3. Salvar Configurações", divider="rainbow")

col1_btn, col2_btn, col3_btn = st.columns([1, 2, 1])
with col2_btn:
    if st.button("💾 Salvar Configurações para Avaliação", type="primary", disabled=not selected_models):
        try:
            # 1. Salva os parâmetros globais
            st.session_state.evaluation_params = global_params
            st.session_state.models_to_run = selected_models
            
            # 2. Constrói os modelos e salva no estado
            models_dict, log_msg = get_models(
                schema=stream.schema,
                global_params=global_params,
                models_to_run=selected_models,
                all_model_params=hyperparams
            )
            
            st.session_state.models_to_evaluate = models_dict
            
            st.success("Configurações salvas com sucesso!")
            st.info(f"Parâmetros de delay aplicados: {log_msg}")
            st.info(f"**Próximo Passo:** Vá para a página **'5. Avaliação'** para executar o treinamento.")
            
        except Exception as e:
            st.error(f"Ocorreu um erro ao construir os modelos: {e}")
            st.exception(e)

# Para debug:
# st.divider()
# st.subheader("Debug: Estado da Sessão")
# st.write(st.session_state)