import streamlit as st
import pandas as pd
import os
from utils.style import load_custom_css
from utils.training import get_models 
from capymoa.stream.generator import RandomTreeGenerator, RandomRBFGenerator
from capymoa.stream.drift import DriftStream, AbruptDrift, GradualDrift

load_custom_css("style.css")
st.set_page_config(
    page_title="IDS Stream Mining", 
    page_icon="🛡️",
    layout="centered" 
)

st.title("Configuração de Modelos e Dados")
st.header("Fonte de Dados do Stream", divider="rainbow")

data_source = st.radio(
    "Escolha a origem dos dados para o treinamento:",
    ["Usar Dados do Pré-processamento (Real)", "Gerar Stream Sintético com Drift (DriftStream)"],
    index=1
)

stream_ready = False
total_instances = 0

if data_source == "Usar Dados do Pré-processamento (Real)":
    if ('stream_data' in st.session_state and 
        st.session_state.stream_data is not None and 
        st.session_state.get('X_final_df') is not None): 
        
        try:
            total_instances = st.session_state.X_final_df.shape[0]
            st.success(f"✅ Stream Real carregado do passo anterior! ({total_instances:,} instâncias)")
            stream_ready = True
        except Exception as e:
            st.error(f"Erro ao ler dados reais: {e}")
            stream_ready = False
    else:
        st.warning("⚠️ Nenhum stream pré-processado encontrado na memória.")
        st.markdown("""
        **Para utilizar dados reais:**
        1. Vá para a página **'Pré-processamento'** no menu lateral.
        2. Selecione um arquivo, configure e execute o pipeline.
        3. Retorne aqui.
        """)
        stream_ready = False

elif data_source == "Gerar Stream Sintético com Drift (DriftStream)":
    if RandomTreeGenerator is None:
        st.error("A biblioteca `capymoa` não foi importada corretamente.")
    else:
        st.markdown("Construa um cenário complexo de *concept drift* definindo uma sequência de mudanças.")
        
        with st.container(border=True):
            st.subheader("Escolha a Família do Gerador")
            gen_family = st.selectbox("Família de Dados", ["RandomTreeGenerator (Árvores)", "RandomRBF (Centróides)"]) 
            base_params = {}
            
            if gen_family == "RandomTreeGenerator (Árvores)":
                st.info("Gera dados baseados em árvores de decisão aleatórias. O conceito mudará alterando a 'seed' da árvore.")
                c1, c2 = st.columns(2)
                base_params['num_classes'] = c1.number_input("Num. Classes", 2, 10, 2)
                base_params['num_nominals'] = c2.number_input("Num. Atributos Nominais", 0, 20, 5)
                base_params['num_numerics'] = c1.number_input("Num. Atributos Numéricos", 0, 20, 5)
                base_params['tree_seed_start'] = c2.number_input("Seed Inicial da Árvore", 1, 100, 1)

            elif gen_family == "RandomRBF (Centróides)":
                st.info("Gera dados baseados em centróides radiais (RBF). O conceito mudará alterando a 'seed' do modelo.")
                c1, c2 = st.columns(2)
                base_params['num_classes'] = c1.number_input("Num. Classes", 2, 10, 2)
                base_params['num_attributes'] = c2.number_input("Num. Atributos", 2, 50, 10)
                base_params['num_centroids'] = c1.number_input("Num. Centróides", 10, 100, 50)
                base_params['model_seed_start'] = c2.number_input("Seed Inicial do Modelo", 1, 100, 1)

            st.subheader("Defina a Linha do Tempo (Drifts)")
            if 'drift_data_editor' not in st.session_state:
                 st.session_state.drift_data_editor = pd.DataFrame([
                    {"Posição (Instância)": 5000, "Tipo": "Abrupto", "Largura (Width)": 1},
                    {"Posição (Instância)": 10000, "Tipo": "Gradual", "Largura (Width)": 1000},
                ])

            edited_drifts = st.data_editor(
                st.session_state.drift_data_editor, 
                num_rows="dynamic", 
                column_config={
                    "Posição (Instância)": st.column_config.NumberColumn(min_value=100, step=100, help="Em qual instância o drift começa."),
                    "Tipo": st.column_config.SelectboxColumn(options=["Abrupto", "Gradual"], required=True),
                    "Largura (Width)": st.column_config.NumberColumn(min_value=1, help="1 para Abrupto. Valores maiores (ex: 1000) para Gradual.")
                },
                width='stretch'
            )

            st.subheader("Tamanho Final do Stream")
            total_stream_size = st.number_input(
                "Quantidade Total de Amostras (Instâncias)",
                min_value=1000,
                value=20000,
                step=1000,
                help="Define o tamanho total do stream sintético gerado. Certifique-se de que seja maior que a posição do último drift."
            )
            
            if st.button("🛠️ Gerar Stream Sintético", type="primary", width='stretch'):
                try:
                    sorted_drifts = edited_drifts.sort_values(by="Posição (Instância)")
                    
                    if not sorted_drifts.empty:
                        last_drift_pos = sorted_drifts["Posição (Instância)"].max()
                        if total_stream_size <= last_drift_pos:
                            st.error(f"Erro: O tamanho total ({total_stream_size}) deve ser maior que a posição do último drift ({last_drift_pos}). Aumente o tamanho total.")
                            st.stop()

                    stream_components = []
                    
                    if gen_family.startswith("RandomTree"):
                        current_seed = base_params['tree_seed_start']
                        stream_components.append(RandomTreeGenerator(
                            tree_random_seed=current_seed,
                            num_classes=base_params['num_classes'],
                            num_nominals=base_params['num_nominals'],
                            num_numerics=base_params['num_numerics']
                        ))
                    elif gen_family.startswith("RandomRBF"):
                        current_seed = base_params['model_seed_start']
                        stream_components.append(RandomRBFGenerator(
                            model_random_seed=current_seed,
                            number_of_classes=base_params['num_classes'],
                            number_of_attributes=base_params['num_attributes'],
                            number_of_centroids=base_params['num_centroids']
                        ))

                    last_pos = 0
                    for index, row in sorted_drifts.iterrows():
                        pos = int(row["Posição (Instância)"])
                        drift_type = row["Tipo"]
                        width = int(row["Largura (Width)"])
                        
                        if pos <= last_pos:
                            st.error(f"Erro: A posição {pos} deve ser maior que a anterior {last_pos}.")
                            raise ValueError("Ordem de drift inválida.")
                        
                        if drift_type == "Abrupto":
                            stream_components.append(AbruptDrift(position=pos))
                        else:
                            stream_components.append(GradualDrift(position=pos, width=width))
                        
                        if gen_family.startswith("RandomTree"):
                            current_seed += 1
                            stream_components.append(RandomTreeGenerator(
                                tree_random_seed=current_seed,
                                num_classes=base_params['num_classes'],
                                num_nominals=base_params['num_nominals'],
                                num_numerics=base_params['num_numerics']
                            ))
                        elif gen_family.startswith("RandomRBF"):
                            current_seed += 1
                            stream_components.append(RandomRBFGenerator(
                                model_random_seed=current_seed,
                                number_of_classes=base_params['num_classes'],
                                number_of_attributes=base_params['num_attributes'],
                                number_of_centroids=base_params['num_centroids']
                            ))
                        
                        last_pos = pos

                    synthetic_stream = DriftStream(stream=stream_components)
                    st.session_state.stream_data = synthetic_stream
                    st.session_state.synthetic_max_instances = total_stream_size
                    st.session_state.X_final_df = None 
                    st.success(f"✅ Stream '{gen_family}' criado com sucesso! Tamanho: {total_stream_size}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro ao construir stream: {e}")

        if 'stream_data' in st.session_state and st.session_state.stream_data is not None:
             if st.session_state.get('X_final_df') is None:
                 total_instances = st.session_state.get('synthetic_max_instances', 15000)
                 st.success(f"✅ Stream Sintético Ativo (Tamanho definido: {total_instances})")
                 stream_ready = True



st.header("Parâmetros Globais de Avaliação", divider="rainbow")
safe_max_instances = total_instances if total_instances > 0 else 15000

global_params = {}
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        global_params["MAX_INSTANCES"] = st.number_input(
            "Máximo de Instâncias (MAX_INSTANCES)",
            min_value=1000,
            value=min(20000, safe_max_instances), 
            max_value=safe_max_instances if stream_ready else None,
            step=1000,
            help="Quantas instâncias serão processadas.",
            disabled=not stream_ready
        )
    with c2:
        global_params["WINDOW_SIZE"] = st.number_input(
            "Janela de Avaliação (WINDOW_SIZE)",
            min_value=100,
            value=500,
            step=100,
            help="Janela deslizante para calcular a acurácia.",
            disabled=not stream_ready
        )
    
    c3, c4 = st.columns(2)
    with c3:
        delay_value = st.number_input(
            "Atraso de Rótulo (DELAY_LENGTH)",
            min_value=0,
            value=0,
            step=10,
            help="Atraso para disponibilizar o rótulo. 0 = Sem atraso.",
            disabled=not stream_ready
        )
        global_params["DELAY_LENGTH"] = delay_value if delay_value > 0 else None
        
    with c4:
        global_params["LABEL_PROBABILITY"] = st.slider(
            "Probabilidade de Rótulo (LABEL_PROBABILITY)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Simula perda de rótulos.",
            disabled=not stream_ready
        )

st.header("Seleção e Configuração dos Modelos", divider="rainbow")
st.markdown("Configure os algoritmos de aprendizado e detecção.")

all_model_names = [
    "LeveragingBagging", 
    "HoeffdingAdaptiveTree", 
    "AdaptiveRandomForest",
    "HoeffdingTree"
]

selected_models = st.multiselect(
    "Selecione os modelos:",
    options=all_model_names,
    default=all_model_names[0] 
)

hyperparams = {}

if not selected_models:
    st.warning("Selecione pelo menos um modelo.")
else:
    tabs = st.tabs(selected_models)
    
    for i, model_name in enumerate(selected_models):
        with tabs[i]:
            hyperparams[model_name] = {}
            
            if model_name == "LeveragingBagging":
                c1, c2 = st.columns(2)
                hyperparams[model_name]["ensemble_size"] = c1.number_input("Tamanho do Ensemble", 1, 500, 100, 10, key=f"{model_name}_ens")
                hyperparams[model_name]["random_seed"] = c2.number_input("Random Seed", 1, 100, 1, key=f"{model_name}_rs")
            
            elif model_name == "HoeffdingAdaptiveTree":
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["grace_period"] = c1.number_input("Período de Graça", 10, 1000, 200, 10, key=f"{model_name}_gp")
                hyperparams[model_name]["confidence"] = c2.number_input("Confiança", 0.0, 1.0, 0.01, 0.01, format="%.3f", key=f"{model_name}_conf")
                hyperparams[model_name]["tie_threshold"] = c3.number_input("Limiar Empate", 0.0, 1.0, 0.05, 0.01, format="%.3f", key=f"{model_name}_tie")
                hyperparams[model_name]["leaf_prediction"] = st.selectbox("Preditor Folha", ['NaiveBayes', 'NaiveBayesAdaptive', 'MC'], 1, key=f"{model_name}_leaf")
                hyperparams[model_name]["nb_threshold"] = st.number_input("Limiar NaiveBayes", 0, 100, 0, key=f"{model_name}_nb")

            elif model_name == "AdaptiveRandomForest":
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["ensemble_size"] = c1.number_input("Tamanho do Ensemble", 1, 500, 100, 10, key=f"{model_name}_ens")
                hyperparams[model_name]["max_features"] = c2.number_input("Max Features", 0.1, 1.0, 0.6, 0.1, key=f"{model_name}_maxf")
                hyperparams[model_name]["lambda_param"] = c3.number_input("Lambda", 1.0, 20.0, 6.0, 0.5, key=f"{model_name}_lambda")
                hyperparams[model_name]["disable_drift_detection"] = st.checkbox("Desabilitar Drift Interno", False, key=f"{model_name}_drift")

            elif model_name == "HoeffdingTree":
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["grace_period"] = c1.number_input("Período de Graça", 10, 1000, 200, 10, key=f"{model_name}_gp")
                hyperparams[model_name]["confidence"] = c2.number_input("Confiança", 0.0, 1.0, 0.01, 0.01, format="%.3f", key=f"{model_name}_conf")
                hyperparams[model_name]["tie_threshold"] = c3.number_input("Limiar Empate", 0.0, 1.0, 0.05, 0.01, format="%.3f", key=f"{model_name}_tie")
                hyperparams[model_name]["leaf_prediction"] = st.selectbox("Preditor Folha", ['NaiveBayes', 'MC'], 0, key=f"{model_name}_leaf")
                hyperparams[model_name]["nb_threshold"] = st.number_input("Limiar NaiveBayes", 0, 100, 0, key=f"{model_name}_nb")

            with st.expander("Configurar Detectores de Drift Externos"):
                st.caption("Estes detectores rodarão em paralelo.")
                c1, c2, c3 = st.columns(3)
                hyperparams[model_name]["ddm_min_instances"] = c1.number_input("DDM: Min Instâncias", 10, 1000, 30, key=f"{model_name}_dmin")
                hyperparams[model_name]["ddm_warning_level"] = c2.number_input("DDM: Warning Level", 1.0, 5.0, 2.0, 0.1, key=f"{model_name}_dwarn")
                hyperparams[model_name]["ddm_out_control_level"] = c3.number_input("DDM: Out Control", 1.0, 5.0, 3.0, 0.1, key=f"{model_name}_dout")
                hyperparams[model_name]["adwin_delta"] = c1.number_input("ADWIN: Delta", 0.001, 1.0, 0.002, 0.001, format="%.4f", key=f"{model_name}_adelta")
                hyperparams[model_name]["abcd_delta_drift"] = c2.number_input("ABCD: Delta Drift", 0.001, 1.0, 0.002, 0.001, format="%.4f", key=f"{model_name}_abcd_d")
                hyperparams[model_name]["abcd_delta_warn"] = c3.number_input("ABCD: Delta Warn", 0.001, 1.0, 0.01, 0.01, format="%.3f", key=f"{model_name}_abcd_w")


st.header("Salvar Configurações", divider="rainbow")
col1_btn, col2_btn, col3_btn = st.columns([1, 2, 1])
with col2_btn:
    if st.button("Salvar e Ir para Avaliação", type="primary", disabled=not (stream_ready and selected_models)):
        if not stream_ready:
            st.error("Não é possível salvar: Nenhum stream de dados válido (Real ou Sintético) foi carregado.")
        else:
            try:
                st.session_state.evaluation_params = global_params
                st.session_state.models_to_run = selected_models
                
                stream_schema = st.session_state.stream_data.schema
                
                models_dict, log_msg = get_models(
                    schema=stream_schema,
                    global_params=global_params,
                    models_to_run=selected_models,
                    all_model_params=hyperparams
                )
                
                st.session_state.models_to_evaluate = models_dict
                
                st.success("Configurações salvas com sucesso!")
                if data_source == "Gerar Stream Sintético com Drift (DriftStream)":
                     st.info("Modo: Stream Sintético.")
                else:
                     st.info("Modo: Stream Real (Pré-processado).")
                
                st.info(f"**Próximo Passo:** Vá para a página **'Avaliação'** para executar o treinamento.")
                
            except Exception as e:
                st.error(f"Ocorreu um erro ao construir os modelos: {e}")
                st.exception(e)