import streamlit as st
import pandas as pd
import os
import altair as alt 
from utils.style import load_custom_css
from utils.preprocessing import create_stream_pipeline
load_custom_css("style.css")

st.set_page_config(
    page_title="IDS Stream Mining", 
    page_icon="🛡️",
    layout="centered" 
)

@st.cache_data
def load_sample_df(filepath):
    try:
        df_sample = pd.read_csv(filepath, nrows=50)
        df_sample.columns = df_sample.columns.str.strip()
        numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
        all_cols = df_sample.columns.tolist()
        cols_to_pre_remove = [col for col in all_cols if col not in numeric_cols and col != 'Label'] 
        return df_sample, all_cols, cols_to_pre_remove
    except Exception as e:
        st.error(f"Erro ao ler amostra do arquivo: {e}")
        return None, [], []

def find_default_index(options, default_value):
    try:
        return options.index(default_value)
    except ValueError:
        return 0

# Renderização da Página
st.title("Pré-processamento e Criação do Stream")

# Verificação de Arquivo 
filepath = st.session_state.get('file_to_analyze')
all_cols, cols_to_pre_remove = [], []
file_selected = False
placeholder_options = ['Selecione um arquivo na Base de Dados'] 

if filepath and os.path.exists(filepath):
    df_sample, all_cols, cols_to_pre_remove = load_sample_df(filepath)
    if df_sample is not None:
        file_selected = True 
    else:
        st.error(f"Erro ao ler o arquivo selecionado: {filepath}")
else:
    if not filepath:
        st.warning("NOTA: Nenhum arquivo de dados selecionado. Por favor, vá para a página **'Base de Dados'** e selecione um arquivo no **'Seleção dos Dados'** para habilitar esta página.")
    else:
        st.error(f"Arquivo selecionado '{filepath}' não foi encontrado. Retorne à página anterior e selecione um arquivo válido.")

# Configuração dos Parâmetros 
st.header("Configuração do Pipeline", divider="rainbow")
st.markdown("Defina os parâmetros para limpar os dados e criar o *stream* de dados para o treinamento. As opções ficarão habilitadas assim que um arquivo válido for selecionado na Base de Dados.")

with st.container(border=True):
    st.subheader("Definição das Colunas Principais")
    st.markdown("Defina as colunas essenciais para o modelo: o que ele deve prever (Alvo) e, a ordem em que os dados chegaram (Timestamp).")
    
    label_idx = find_default_index(all_cols, 'Label')
    target_col = st.selectbox(
        "Selecione a Coluna Alvo (Label)", 
        options=all_cols if file_selected else placeholder_options, 
        index=label_idx if file_selected else 0,
        help="Esta é a coluna que o modelo tentará prever (ex: 'Label', 'Attack_Type').",
        disabled=not file_selected
    )
    
    ts_idx = find_default_index(all_cols, 'Timestamp')
    timestamp_col = st.selectbox(
        "Selecione a Coluna para ordenação (Timestamp)", 
        options=['Nenhuma'] + (all_cols if file_selected else []), 
        index=ts_idx + 1 if file_selected and ts_idx >= 0 else 0,
        help="Se selecionado, os dados serão ordenados por esta coluna para simular um stream em ordem cronológica. Se 'Nenhuma', a ordem do CSV será usada.",
        disabled=not file_selected
    )
    timestamp_col = None if timestamp_col == 'Nenhuma' else timestamp_col

with st.container(border=True):
    st.subheader("Limpeza de Dados e Imputação")
    st.markdown("Defina como o pipeline deve tratar dados ausentes, infinitos ou colunas irrelevantes.")

    available_cols = [col for col in all_cols if col != target_col and col != timestamp_col]
    cols_to_pre_remove_default = [col for col in cols_to_pre_remove if col in available_cols]

    cols_to_remove = st.multiselect(
        "Colunas para Remover (Pré-filtragem)",
        options=available_cols if file_selected else placeholder_options,
        default=cols_to_pre_remove_default if file_selected else [],
        help="Colunas que devem ser removidas ANTES da seleção de features (Ex: IDs, IPs, ou colunas não-numéricas).",
        disabled=not file_selected
    )
    
    imputation_method = st.selectbox(
        "Método de Imputação (para Nulos/Infinitos)",
        options=['Mediana', 'Média', 'Preencher com 0', 'Remover Linhas'],
        index=0,
        help="Como o pipeline deve tratar células vazias (NaN) ou infinitas (inf) nos dados numéricos.",
        disabled=not file_selected
    )

with st.container(border=True):
    st.subheader("Seleção de Features")
    st.markdown("""
    Após a limpeza, podemos reduzir ainda mais o número de colunas (features) para acelerar o treinamento e, potencialmente, melhorar a precisão.
    """)
    
    feature_selection_method = st.radio(
        "Escolha o método de seleção de features:",
        ['Seleção Manual', 'Seleção Automática'],
        index=0,
        horizontal=True,
        disabled=not file_selected
    )
    
    available_features = [col for col in available_cols if col not in cols_to_remove]
    
    # --- Valores Padrão ---
    manual_features_list = []
    auto_algo = 'Random Forest Importance'
    n_features_auto = 10
    rf_n_estimators = 100
    rf_iterations = 1
    rf_max_depth = None
    rf_min_samples_leaf = 1
    use_max_depth_none = True
    skb_score_func_name = 'f_classif'
    pca_svd_solver = 'auto'
    pca_whiten = False

    if feature_selection_method == 'Seleção Manual':
        st.markdown("Selecione manualmente as features que você deseja manter. **Se este campo ficar vazio, todas as features restantes serão usadas.**")
        manual_features_list = st.multiselect(
            "Manter APENAS estas features:",
            options=available_features if file_selected else placeholder_options,
            default=[],
            help="Se você preencher este campo, o pipeline irá descartar TODAS as colunas, exceto as que você selecionar aqui.",
            disabled=not file_selected
        )
    
    elif feature_selection_method == 'Seleção Automática':
        st.markdown("Escolha um algoritmo para pontuar e selecionar as melhores features automaticamente.")
        auto_algo = st.selectbox(
            "Algoritmo de Seleção/Extração",
            options=[
                'Random Forest Importance', 
                'SelectKBest', 
                'PCA (Extração de Componentes)'
            ],
            index=0,
            disabled=not file_selected
        )
        
        if auto_algo == 'Random Forest Importance':
            st.markdown("##### Hiperparâmetros do Random Forest")
            rf_iterations = st.number_input(
                "Número de Iterações",
                min_value=1, max_value=10, value=1, step=1,
                disabled=not file_selected,
                help="Rodar o algoritmo N vezes e tirar a média das importâncias. Aumenta a estabilidade, mas também o tempo de processamento."
            )
            rf_n_estimators = st.number_input(
                "Número de Árvores por Iteração (n_estimators)",
                min_value=10, max_value=1000, value=100, step=10,
                disabled=not file_selected,
                help="Número de árvores na floresta. Valores maiores são mais precisos, mas levam mais tempo."
            )
            use_max_depth_none = st.checkbox("Usar max_depth=None (sem limite)", value=True, disabled=not file_selected)
            rf_max_depth_value = st.number_input(
                "Profundidade Máxima da Árvore (max_depth)",
                min_value=1, value=10, step=1,
                disabled=use_max_depth_none or not file_selected,
                help="Limite a profundidade de cada árvore. Desmarque o checkbox acima para definir um limite."
            )
            rf_max_depth = None if use_max_depth_none else rf_max_depth_value
            rf_min_samples_leaf = st.number_input(
                "Mínimo de Amostras por Folha (min_samples_leaf)",
                min_value=1, value=1, step=1,
                disabled=not file_selected,
                help="O número mínimo de amostras necessário para ser um nó folha."
            )
        
        elif auto_algo == 'SelectKBest':
            st.markdown("##### Hiperparâmetros do SelectKBest")
            skb_score_func_str = st.selectbox(
                "Função de Pontuação (score_func)",
                options=["ANOVA (f_classif)", "Informação Mútua (mutual_info_classif)"],
                index=0,
                disabled=not file_selected,
                help="O teste estatístico usado para pontuar as features. 'ANOVA' é mais rápido, 'Informação Mútua' pode capturar relações não-lineares."
            )
            skb_score_func_name = skb_score_func_str.split(' ')[1].replace('(', '').replace(')', '')
        
        elif auto_algo == 'PCA (Extração de Componentes)':
            st.markdown("##### Hiperparâmetros do PCA")
            pca_svd_solver = st.selectbox(
                "SVD Solver (svd_solver)",
                options=['auto', 'full', 'randomized'],
                index=0,
                disabled=not file_selected,
                help="O método que o PCA usa para decompor os dados. 'randomized' costuma ser mais rápido em datasets grandes."
            )
            pca_whiten = st.checkbox("Normalizar Componentes (whiten=True)", value=False, disabled=not file_selected,
                                     help="Se marcado, normaliza os componentes resultantes. Pode ser útil para alguns algoritmos.")
            
        n_features_auto = st.number_input(
            f"Número de features/componentes a manter:",
            min_value=1,
            max_value=len(available_features) if file_selected and available_features else 1,
            value=min(10, len(available_features)) if available_features else 1, 
            step=1,
            disabled=not file_selected
        )

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button_clicked = st.button(
        "🚀 Iniciar Pré-processamento e Criar Stream", 
        type="primary", 
        disabled=not file_selected
    )

if start_button_clicked:
    st.session_state.target_col = target_col
    st.session_state.timestamp_col = timestamp_col
    st.session_state.cols_to_remove = cols_to_remove
    st.session_state.imputation_method = imputation_method
    
    if feature_selection_method == 'Seleção Manual':
        st.session_state.feature_selection_method = 'Seleção Manual'
    else:
        st.session_state.feature_selection_method = auto_algo
        
    st.session_state.n_features_auto = n_features_auto
    st.session_state.manual_features_list = manual_features_list
    st.session_state.rf_n_estimators = rf_n_estimators
    st.session_state.rf_iterations = rf_iterations
    st.session_state.rf_max_depth = rf_max_depth
    st.session_state.rf_min_samples_leaf = rf_min_samples_leaf
    st.session_state.skb_score_func_name = skb_score_func_name
    st.session_state.pca_svd_solver = pca_svd_solver
    st.session_state.pca_whiten = pca_whiten
    
    log_placeholder = st.empty() 
    
    with st.spinner("Executando pipeline de pré-processamento... Isso pode levar alguns minutos."):
        stream, le, X_data_df_cleaned, df_processed, log_messages, feature_report = create_stream_pipeline(
            file_path=filepath,
            target_label_col=target_col,
            timestamp_col=timestamp_col,
            cols_para_remover=cols_to_remove,
            imputation_method=imputation_method,
            feature_selection_method=st.session_state.feature_selection_method, 
            n_features_auto=n_features_auto,
            manual_features_list=manual_features_list,
            n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_min_samples_leaf=rf_min_samples_leaf,
            rf_iterations=rf_iterations,
            skb_score_func_name=skb_score_func_name,
            pca_svd_solver=pca_svd_solver,
            pca_whiten=pca_whiten
        )
    
    log_placeholder.text_area("Logs do Processamento", "\n".join(log_messages), height=300)
    
    if stream:
        st.success("Pipeline executado com sucesso! O Stream está pronto.")
        
        st.session_state.stream_data = stream
        st.session_state.label_encoder = le
        st.session_state.df_processed = df_processed 
        st.session_state.X_final_df = X_data_df_cleaned 
        st.session_state.feature_importance_report = feature_report
        
        st.header("Resultado do Pipeline", divider="rainbow")
        st.subheader("Análise Pós-Processamento")
        
        if st.session_state.feature_importance_report:
            if st.session_state.feature_selection_method == 'PCA (Extração de Componentes)':
                chart_title = f"Variância Explicada (Top {st.session_state.n_features_auto} Componentes)"
                score_title = "Variância Explicada"
            else:
                chart_title = f"Importância das Features (Top {st.session_state.n_features_auto})"
                score_title = "Score de Importância"

            with st.expander(chart_title, expanded=True):
                report_dict = st.session_state.feature_importance_report
                df_importance = pd.DataFrame(
                    list(report_dict.items()), 
                    columns=['Feature', 'Score']
                ).sort_values(by='Score', ascending=False)
                
                df_importance_top = df_importance.head(st.session_state.n_features_auto)
                
                chart = alt.Chart(df_importance_top).mark_bar().encode(
                    x=alt.X('Score:Q', title=score_title),
                    y=alt.Y('Feature:N', sort='-x'), 
                    tooltip=['Feature', 'Score']
                ).interactive()
                
                st.altair_chart(chart, width='stretch')
        
        final_features = X_data_df_cleaned.columns.tolist()
        with st.expander(f"Lista Final de Features ({len(final_features)})", expanded=False):
            st.code(f"{final_features}")
            
        st.markdown("##### Distribuição de Classes")
        report_df = df_processed.loc[X_data_df_cleaned.index][target_col].value_counts().reset_index()
        report_df.columns = [target_col, 'Contagem']
        
        bar_chart = alt.Chart(report_df).mark_bar().encode(
            x=alt.X(target_col, sort=None),
            y=alt.Y('Contagem'),
            color=alt.Color(target_col, legend=alt.Legend(title="Legenda", orient='right')),
            tooltip=[target_col, 'Contagem']
        ).interactive()
        st.altair_chart(bar_chart, width='stretch')
        
        if timestamp_col:
            st.markdown("##### Distribuição de Ataques ao Longo do Tempo")
            
            try:
                df_plot = df_processed.copy()
                df_plot['time_bin'] = df_plot[timestamp_col].dt.floor('T')
                df_agg = df_plot.groupby(['time_bin', target_col]).size().reset_index(name='Contagem')
                
                area_chart = alt.Chart(df_agg).mark_area().encode(
                    x=alt.X('time_bin', title="Timestamp", axis=alt.Axis(format="%H:%M")),
                    y=alt.Y('Contagem', stack='zero'), 
                    color=alt.Color(target_col, legend=alt.Legend(title="Legenda", orient='right')),
                    tooltip=[alt.Tooltip('time_bin', format="%H:%M"), target_col, 'Contagem']
                ).interactive()
                
                st.altair_chart(area_chart, width='stretch')
            except Exception as e:
                st.warning(f"Não foi possível gerar o gráfico de distribuição ao longo do tempo: {e}")
            
        
        st.info("**Próximo Passo:** Os dados processados e o *stream* foram salvos na sessão. Clique em **'Modelos'** na barra lateral para continuar.")
        
    else:
        st.error("Ocorreu um erro durante o processamento. Verifique os logs acima para mais detalhes.")