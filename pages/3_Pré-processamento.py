# pages/3_Pré-processamento.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
# Importa a função de pré-processamento do seu arquivo utils
# OBS: O conteúdo da função 'perform_preprocessing' em utils/preprocessing.py
# deve ser ajustado para receber as colunas selecionadas e o método de imputação.

# Assumindo que o arquivo original está na chave 'df_original' e que as colunas
# foram limpas (minúsculas, sem espaços) na página 'Base de Dados'.

# --- Constantes para Nomes de Colunas ---
# Usaremos nomes minúsculos e sem espaço, conforme a limpeza feita na página 2
TARGET_COL_DEFAULT = 'label'
TIMESTAMP_COL_DEFAULT = 'timestamp'


# Função de Pré-processamento (Esta função deve refletir a lógica de 'criar_stream'
# e seria idealmente colocada em utils/preprocessing.py, mas a lógica de UI está aqui)
def perform_stream_preprocessing(df, target_col, timestamp_col, cols_para_remover, imputation_method, features_selecionadas=None):
    """Simula a lógica do pipeline 'criar_stream' para preparar o DataFrame."""
    
    # 1. Copia e Garante Nomes Limpos (já deve vir limpo, mas é uma segurança)
    df_processed = df.copy()
    df_processed.columns = [col.strip().lower() for col in df_processed.columns]
    
    # 2. Tratar Infinitos
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 3. Ordenar por Timestamp
    if timestamp_col in df_processed.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Força coerção de erro: datas inválidas viram NaT
            df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce') 
        
        if not df_processed[timestamp_col].isnull().all():
            df_processed.sort_values(by=timestamp_col, inplace=True)
            df_processed.reset_index(drop=True, inplace=True)

    # 4. Limpeza e Preparação de X/y
    
    # Separar Target (y)
    if target_col not in df_processed.columns:
        st.error(f"Erro: Coluna de rótulo '{target_col}' não encontrada após a limpeza.")
        return None, None
        
    y_data_df = df_processed[target_col]
    
    # Remover colunas desnecessárias (target, timestamp e colunas a remover)
    todas_cols_para_remover = [target_col, timestamp_col] + [col.strip().lower() for col in cols_para_remover]
    cols_existentes_para_remover = [col for col in todas_cols_para_remover if col in df_processed.columns]
    
    X_data_df = df_processed.drop(columns=cols_existentes_para_remover, errors='ignore')

    # Garantir X numérico
    X_data_df_numeric = X_data_df.select_dtypes(include=np.number)
    non_numeric_cols = X_data_df.select_dtypes(exclude=np.number).columns.tolist()
    
    if non_numeric_cols:
        st.warning(f"Removendo {len(non_numeric_cols)} colunas não numéricas que sobraram (ex: {non_numeric_cols[:3]}).")
    
    # 5. Imputar NaNs
    nan_counts = X_data_df_numeric.isnull().sum().sum()
    X_data_df_cleaned = X_data_df_numeric.copy()
    
    if nan_counts > 0:
        if imputation_method == 'Mediana':
            X_data_df_cleaned = X_data_df_numeric.fillna(X_data_df_numeric.median()).fillna(0)
        elif imputation_method == 'Média':
            X_data_df_cleaned = X_data_df_numeric.fillna(X_data_df_numeric.mean()).fillna(0)
        # Adicionei Moda e Constante aqui para evitar erros de referência, mas mantive a lógica de média/mediana
        elif imputation_method == 'Moda':
            X_data_df_cleaned = X_data_df_numeric.fillna(X_data_df_numeric.mode().iloc[0]).fillna(0)
        elif imputation_method == 'Valor Constante (0)':
            X_data_df_cleaned = X_data_df_numeric.fillna(0)
            
        st.info(f"Imputados {nan_counts} valores nulos/infinitos com **{imputation_method}**.")
    else:
        st.info("Nenhum valor nulo/infinito encontrado nas colunas numéricas.")

    # 6. Seleção de Features Finais
    if features_selecionadas:
        features_selecionadas_clean = [col.strip().lower() for col in features_selecionadas]
        features_existentes = [col for col in features_selecionadas_clean if col in X_data_df_cleaned.columns]
        
        if features_existentes:
            X_data_df_final = X_data_df_cleaned[features_existentes]
            st.success(f"Seleção de features aplicada: {len(features_existentes)} colunas mantidas.")
        else:
            st.error("Nenhuma das features selecionadas foi encontrada. Usando todas as colunas numéricas limpas.")
            X_data_df_final = X_data_df_cleaned
    else:
        X_data_df_final = X_data_df_cleaned
        st.info("Nenhuma seleção específica de features: usando todas as colunas numéricas limpas.")

    # Retorna DataFrame X final e a Série y (target)
    return X_data_df_final, y_data_df


# ----------------------------------------------------------------------
# --- INTERFACE STREAMLIT ---
# ----------------------------------------------------------------------

st.title("⚙️ Pré-processamento do Dataset")
st.write("Configure os passos de limpeza e seleção de features antes de criar o stream.")

# Verifica se os dados originais foram carregados
df_original_data = st.session_state.get('df_original')
if df_original_data is None:
    st.warning("⚠️ Por favor, primeiro selecione e carregue um dataset na aba **'Base de Dados'**.")
    st.stop()

# Assume que o df_original é o DataFrame limpo (colunas em minúsculas e sem espaços)
df = df_original_data.copy()
all_cols = list(df.columns)

st.info(f"Dataset carregado: **{st.session_state.selected_csv_name}** ({df.shape[0]} linhas).")

# --- 1. CONFIGURAÇÃO DE COLUNAS ESSENCIAIS ---
st.subheader("1. Configurações Essenciais")

col_target, col_timestamp = st.columns(2)

# --- INÍCIO DA ALTERAÇÃO SOLICITADA: TROCA st.text_input por st.selectbox ---

# Coluna de Rótulo (Target) - AGORA COM SELECTBOX
with col_target:
    default_target = st.session_state.get('target_col', TARGET_COL_DEFAULT)
    default_index_target = all_cols.index(default_target) if default_target in all_cols else 0
    
    target_col = st.selectbox("Nome da Coluna de Rótulo (Target)", 
                              options=all_cols,
                              index=default_index_target)
    st.session_state.target_col = target_col # Salva para reuso

# Coluna de Timestamp (Ordenação) - AGORA COM SELECTBOX
with col_timestamp:
    default_timestamp = st.session_state.get('timestamp_col', TIMESTAMP_COL_DEFAULT)
    timestamp_options = [None] + all_cols # Inclui None para não ordenar

    if default_timestamp is None:
        default_index_timestamp = 0
    elif default_timestamp in all_cols:
        default_index_timestamp = all_cols.index(default_timestamp) + 1 # +1 devido ao None no início
    else:
        default_index_timestamp = 0
    
    timestamp_col = st.selectbox("Coluna para ORDENAÇÃO (Timestamp)", 
                                 options=timestamp_options,
                                 index=default_index_timestamp,
                                 help="Opcional. O DataFrame será ordenado por esta coluna.")
    st.session_state.timestamp_col = timestamp_col # Salva para reuso

# --- FIM DA ALTERAÇÃO SOLICITADA ---


# --- 2. SELEÇÃO E REMOÇÃO DE COLUNAS ---
st.subheader("2. Seleção de Features")

col_keep, col_remove = st.columns(2)

# Colunas disponíveis para manipulação (exclui Target e Timestamp)
excluded_from_available = [target_col]
if timestamp_col:
    excluded_from_available.append(timestamp_col)
    
all_available_cols = [c for c in all_cols if c not in excluded_from_available]

# Colunas a Remover (não-features)
with col_remove:
    # Filtra colunas que não são target ou timestamp, para sugerir remoção
    removable_cols_candidates = all_available_cols
    
    # Blinda o default para evitar StreamlitAPIException (mantendo a lógica de persistência)
    initial_cols_to_remove = st.session_state.get('cols_to_remove', [])
    cols_to_remove_defaults = [c for c in initial_cols_to_remove if c in removable_cols_candidates]
    
    cols_to_remove = st.multiselect("Colunas a REMOVER (ex: IDs, IPs)", 
                                    options=removable_cols_candidates,
                                    default=cols_to_remove_defaults) # Usa os defaults filtrados
    st.session_state.cols_to_remove = cols_to_remove

# Colunas a Manter (Seleção de Features Opcional)
with col_keep:
    # Colunas que sobrariam após a remoção básica: LÓGICA DE EXCLUSÃO CRUZADA RESTAURADA
    remaining_cols_candidates = [c for c in all_available_cols if c not in cols_to_remove]
    
    # Blinda o default para evitar StreamlitAPIException
    initial_features_to_keep = st.session_state.get('features_to_keep', [])
    features_to_keep_defaults = [c for c in initial_features_to_keep if c in remaining_cols_candidates]
    
    # Padrão: Se não há persistência válida, usa TODAS as remanescentes como padrão (lógica inicial)
    default_features = features_to_keep_defaults or remaining_cols_candidates
    
    features_to_keep = st.multiselect("Features a MANTER (Seleção de features)", 
                                      options=remaining_cols_candidates, # Lista que exclui as removidas
                                      default=default_features,
                                      help="Selecione um subconjunto de features numéricas para o modelo. Se vazio, todas as colunas numéricas remanescentes serão usadas.")
    st.session_state.features_to_keep = features_to_keep

# --- AVISO SOBRE DADOS NÃO USADOS ---
features_not_kept = [c for c in remaining_cols_candidates if c not in features_to_keep]
if features_not_kept:
    st.markdown("##### ⚠️ Aviso de Features Não Usadas:")
    st.warning(f"As features desmarcadas (`{', '.join(features_not_kept)}`) **NÃO** serão incluídas no conjunto final de Features (X).")
else:
    st.info("Todas as features remanescentes foram selecionadas para manter.")

final_cols_to_remove = cols_to_remove 


# --- 3. TRATAMENTO DE NULOS (IMPUTAÇÃO) ---
st.subheader("3. Tratamento de Valores Ausentes (NaN)")

# Mediana é geralmente a mais robusta para datasets com outliers (comuns em tráfego de rede)
imputation_method = st.selectbox("Método de Imputação de NaN/Infinito", 
                                 options=['Mediana', 'Média', 'Moda', 'Valor Constante (0)'],
                                 index=['Mediana', 'Média', 'Moda', 'Valor Constante (0)'].index(st.session_state.get('imputation_method', 'Mediana')),
                                 help="A Mediana é mais robusta contra outliers.")
st.session_state.imputation_method = imputation_method


# --- 4. BOTÃO DE PRÉ-PROCESSAMENTO ---
st.markdown("---")
if st.button("Aplicar Configurações e Preparar Stream", type="primary"):
    
    # Validação Mínima
    if not target_col or target_col not in all_cols:
        st.error(f"A coluna de rótulo '{target_col}' não foi encontrada ou está vazia. Verifique o nome.")
        st.stop()
        
    if timestamp_col and timestamp_col not in all_cols:
        st.warning(f"A coluna de tempo '{timestamp_col}' não foi encontrada.")

    with st.spinner("Criando o DataFrame de Features (X) e Rótulos (y)..."):
        
        # Chama a função de pré-processamento
        X_data_df, y_data_series = perform_stream_preprocessing(
            df=df,
            target_col=target_col,
            timestamp_col=timestamp_col,
            cols_para_remover=final_cols_to_remove,
            imputation_method=imputation_method,
            features_selecionadas=features_to_keep
        )

    if X_data_df is not None and y_data_series is not None:
        st.session_state.X_features = X_data_df 
        st.session_state.y_target = y_data_series 
        
        # Armazenar o DataFrame 'df_processed' para visualização
        df_processed_preview = X_data_df.copy()
        df_processed_preview[target_col] = y_data_series.values
        st.session_state.df_processed = df_processed_preview
        
        st.success("Pré-processamento de Features e Rótulos concluído!")
        
        st.subheader("Features Prontas (X)")
        st.info(f"O conjunto de features (X) final tem {X_data_df.shape[0]} amostras e {X_data_df.shape[1]} features.")
        st.dataframe(X_data_df.head(), width='stretch')
        
        st.subheader("Rótulos Prontos (y)")
        st.info(f"O conjunto de rótulos (y) final tem {y_data_series.shape[0]} amostras.")
        st.dataframe(pd.DataFrame(y_data_series).head(), width='stretch')

    else:
        st.error("O Pré-processamento falhou. Verifique as colunas essenciais.")

# --- LÓGICA DE EXIBIÇÃO DE DADOS PROCESSADOS (Se já existirem) ---
elif st.session_state.get('df_processed') is not None:
    st.subheader("Dados Atualmente Processados (Prévia)")
    st.info(f"DataFrame processado pronto para a próxima etapa: **{st.session_state.df_processed.shape[0]} amostras**.")
    st.dataframe(st.session_state.df_processed.head(), width='stretch')