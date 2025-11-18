import streamlit as st
import pandas as pd
import altair as alt 
import os
import warnings # Importa módulo de warnings

# --- Request 2: Silenciar Warnings no Terminal ---
warnings.filterwarnings("ignore") # Ignora avisos para manter o terminal limpo

from utils.style import load_custom_css
from utils.evaluation import run_evaluation_stream, get_attack_summary_table 
import math 

# --- Configuração da Página ---
st.set_page_config(
    page_title="Avaliação", 
    page_icon="📊",
    layout="wide" 
)
load_custom_css("style.css")

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# --- Renderização da Página ---
st.title("📊 5. Avaliação dos Modelos")

# --- Passo 1: Verificar se os dados existem ---
if 'stream_data' not in st.session_state or \
   'models_to_evaluate' not in st.session_state or \
   'evaluation_params' not in st.session_state or \
   'df_processed' not in st.session_state: 
    
    st.error("Nenhuma configuração completa de treinamento encontrada.")
    st.warning("Por favor, retorne às páginas anteriores e execute todo o fluxo (Base de Dados -> Pré-processamento -> Modelos) antes de executar a avaliação.")
    st.stop()

# --- Carrega os dados da sessão ---
stream = st.session_state.stream_data
models_to_evaluate = st.session_state.models_to_evaluate
eval_params = st.session_state.evaluation_params
models_to_run = st.session_state.models_to_run
df_processed = st.session_state.df_processed
target_col = st.session_state.target_col

if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# --- Tabela de Ataques ---
# with st.expander("Ver Resumo dos Ataques no Stream"):
#     st.markdown("Esta tabela mostra onde cada ataque (não-BENIGN) começa e termina no *stream* de dados processado.")
#     summary_table = get_attack_summary_table(df_processed, target_col)
#     if summary_table.empty:
#         st.info("Nenhum ataque (não-BENIGN) foi encontrado no stream processado.")
#     else:
#         st.dataframe(summary_table, width='stretch', hide_index=True)

# --- Passo 2: Botão de Execução ---
st.header("Executar Avaliação Prequencial", divider="rainbow")
st.markdown(f"Clique no botão abaixo para iniciar a avaliação de **{len(models_to_run)}** modelo(s) em **{eval_params.get('MAX_INSTANCES'):,}** instâncias.")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button_clicked = st.button("🚀 Iniciar Avaliação do Stream", type="primary")

if start_button_clicked:
    st.session_state.evaluation_results = None
    
    st.markdown("---")
    st.subheader("Resultados em Tempo Real")
    
    progress_text = st.empty()
    progress_bar = st.progress(0, text="Iniciando...")
    
    live_placeholders = {}
    model_chunks = list(chunk_list(models_to_run, 4))
    
    for chunk in model_chunks:
        live_chart_cols = st.columns(4)
        for i, model_name in enumerate(chunk):
            with live_chart_cols[i]:
                st.markdown(f"##### {model_name}")
                live_placeholders[model_name] = st.empty()

    accuracy_history = []
    drift_history = []
    
    stream.restart() 
    total_instances = eval_params.get("MAX_INSTANCES")
    
    for result in run_evaluation_stream(stream, models_to_evaluate, eval_params):
        
        if isinstance(result, dict):
            if result.get("status") == "completed":
                st.session_state.evaluation_results = result 
                break 
            else:
                st.error(result.get("error", "Erro desconhecido no generator de avaliação."))
                break 
        
        metrics_update, instance_count_history = result
        instance_idx = metrics_update.get("instance", 0)
        
        progress_percentage = instance_idx / total_instances
        progress_text.text(f"Processando instância {instance_idx:,} de {total_instances:,} ({progress_percentage:.1%})")
        progress_bar.progress(progress_percentage)
        
        for model_name in models_to_run:
            if model_name in metrics_update:
                metrics = metrics_update[model_name]
                accuracy_history.append({
                    "Instância": instance_idx,
                    "Modelo": model_name,
                    "Acurácia": metrics.get("Acurácia", 0)
                })
                
                # for detector in ["DDM", "ADWIN", "ABCD"]:
                #     drift_key = f"Drift ({detector})"
                #     if metrics.get(drift_key, 0) == 1:
                #         drift_history.append({
                #             "Instância": instance_idx,
                #             "Modelo/Detector": f"{model_name} ({detector})",
                #             "Detector": detector
                #         })

        if accuracy_history:
            df_acc = pd.DataFrame(accuracy_history)
            df_drift = pd.DataFrame(drift_history)
            
            for model_name, placeholder in live_placeholders.items():
                
                df_acc_model = df_acc[df_acc['Modelo'] == model_name]
                
                if not df_drift.empty:
                    df_drift_model = df_drift[df_drift['Modelo/Detector'].str.startswith(model_name)]
                else:
                    df_drift_model = pd.DataFrame(columns=['Instância', 'Modelo/Detector', 'Detector'])
                
                acc_chart = alt.Chart(df_acc_model).mark_line(interpolate='step').encode(
                    x=alt.X('Instância', axis=alt.Axis(format=',d')),
                    y=alt.Y('Acurácia', scale=alt.Scale(domain=[0.0, 1.0])),
                    tooltip=['Instância', 'Acurácia']
                ).interactive()
                
                if not df_drift_model.empty:
                    vlines = alt.Chart(df_drift_model).mark_rule(strokeDash=[5,5]).encode(
                        x='Instância',
                        color=alt.Color('Detector', title="Drift", legend=alt.Legend(orient='bottom')),
                        tooltip=['Instância', 'Detector']
                    )
                    final_chart = acc_chart + vlines
                else:
                    final_chart = acc_chart
                
                placeholder.altair_chart(final_chart, width='stretch')

    progress_text.text(f"Avaliação concluída! Processado {total_instances:,} instâncias.")
    progress_bar.progress(1.0)
    
    if st.session_state.evaluation_results:
        st.success("Avaliação finalizada!")

# --- Passo 3: Exibição dos Resultados (Tabs) ---
if st.session_state.evaluation_results:
    results = st.session_state.evaluation_results
    final_report = results["final_report"]
    models_final_state = results["models_final_state"]
    instance_history = results["instance_history"]
    
    st.markdown("---")
    st.header("Resultados Finais da Avaliação", divider="rainbow")
    
    df_acc_final = pd.DataFrame([
        {"Instância": i, "Modelo": m, "Acurácia": a}
        for m, state in models_final_state.items()
        for i, a in zip(instance_history, state["results_accuracy"])
    ])
    
    df_metrics_final = pd.DataFrame(final_report).T.reset_index()
    df_metrics_final = df_metrics_final.rename(columns={"index": "Modelo"})
    
    # Formata as colunas para 4 casas decimais
    for col in ["Acurácia", "F1-Score", "Precision", "Recall", "Kappa"]:
        if col in df_metrics_final.columns:
            df_metrics_final[col] = pd.to_numeric(df_metrics_final[col], errors='coerce').fillna(0.0)
            df_metrics_final[col] = df_metrics_final[col].map('{:,.4f}'.format)

    tab_list = ["Comparativo Geral"] + models_to_run
    tabs = st.tabs(tab_list)
    
    with tabs[0]:
        st.subheader("Gráfico de Acurácia Comparativa")
        
        acc_chart = alt.Chart(df_acc_final).mark_line(interpolate='step').encode(
            x=alt.X('Instância', axis=alt.Axis(format=',d')),
            y=alt.Y('Acurácia', scale=alt.Scale(domain=[0.0, 1.0])),
            color=alt.Color('Modelo', legend=alt.Legend(orient='bottom')), 
            tooltip=['Instância', 'Modelo', 'Acurácia']
        ).interactive()
        st.altair_chart(acc_chart, width='stretch')
        
        st.subheader("Métricas Cumulativas Finais")
        st.dataframe(df_metrics_final, width='stretch', hide_index=True)

    for i, model_name in enumerate(models_to_run):
        with tabs[i+1]:
            state = models_final_state[model_name]
            
            st.subheader(f"Desempenho: {model_name}")
            
            df_acc_model = df_acc_final[df_acc_final['Modelo'] == model_name]
            drift_points_data = []
            for detector in ["ddm", "adwin", "ABCD"]:
                key = f"results_drift_{detector}"
                for instance in state[key]:
                    drift_points_data.append({"Instância": instance, "Detector": detector.upper()})
            df_drift_points = pd.DataFrame(drift_points_data)
            
            acc_chart_model = alt.Chart(df_acc_model).mark_line(interpolate='step').encode(
                x=alt.X('Instância', axis=alt.Axis(format=',d')),
                y=alt.Y('Acurácia', scale=alt.Scale(domain=[0.0, 1.0])),
                tooltip=['Instância', 'Acurácia']
            ).interactive()
            
            if not df_drift_points.empty:
                vlines = alt.Chart(df_drift_points).mark_rule(strokeDash=[5,5]).encode(
                    x='Instância',
                    color=alt.Color('Detector', title="Drift", legend=alt.Legend(orient='bottom')),
                    tooltip=['Instância', 'Detector']
                )
                st.altair_chart(acc_chart_model + vlines, width='stretch')
            else:
                st.altair_chart(acc_chart_model, width='stretch')

            col_metrics, col_drift_chart = st.columns([1, 1])

            with col_metrics:
                st.subheader("Métricas Cumulativas Finais")
                model_metrics = final_report[model_name]
                model_metrics_float = {k: float(v) for k, v in model_metrics.items()}
                
                st.dataframe(
                    pd.Series(model_metrics_float, name="Score"), 
                    width='stretch',
                    column_config={"Score": st.column_config.NumberColumn(format="%.4f")}
                )

            with col_drift_chart:
                st.subheader("Drifts Detectados (Contagem)")
                drift_counts = [
                    {"Detector": "DDM", "Contagem": len(state['results_drift_ddm'])},
                    {"Detector": "ADWIN", "Contagem": len(state['results_drift_adwin'])},
                    {"Detector": "ABCD", "Contagem": len(state['results_drift_ABCD'])}
                ]
                df_drift_counts = pd.DataFrame(drift_counts)

                bars = alt.Chart(df_drift_counts).mark_bar().encode(
                    x=alt.X('Contagem:Q', title="Total de Drifts"),
                    y=alt.Y('Detector:N', sort=None, title=None),
                    color=alt.Color('Detector', legend=alt.Legend(title="Detector", orient='right'))
                )
                
                text = bars.mark_text(
                    align='left',
                    baseline='middle',
                    dx=3
                ).encode(
                    text=alt.Text('Contagem:Q')
                )
                
                chart = bars + text
                st.altair_chart(chart, width='stretch')