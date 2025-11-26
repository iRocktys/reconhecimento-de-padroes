import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import LabelEncoder
from capymoa.stream import NumpyStream

def create_stream_pipeline(
    file_path, 
    target_label_col, 
    timestamp_col, 
    cols_para_remover, 
    imputation_method='Mediana', 
    feature_selection_method='Seleção Manual', 
    manual_features_list=None,
    correlation_method='pearson',
    correlation_threshold=0.95
):
    log_messages = []
    
    def log(message):
        log_messages.append(message)

    if NumpyStream is None:
        log("❌ ERRO CRÍTICO: A biblioteca 'capymoa' não foi encontrada. Instale-a com 'pip install capymoa'")
        return None, None, None, None, log_messages, None

    feature_report = {}

    try:
        log(f"--- Iniciando Pipeline: {file_path} ---")
        
        log("[Passo 1/7] Carregando arquivo CSV completo...")
        df = pd.read_csv(file_path)
        df_processed = df.copy()
        log(f"    - Arquivo carregado. Shape inicial: {df_processed.shape}")

        log("[Passo 2/7] Limpando nomes das colunas (removendo espaços)...")
        df_processed.columns = df_processed.columns.str.strip()
        target_label_col = target_label_col.strip()
        if timestamp_col:
            timestamp_col = timestamp_col.strip()
        log("    - Colunas limpas.")

        log("[Passo 3/7] Verificando e ordenando por Timestamp...")
        if timestamp_col and timestamp_col in df_processed.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce')
            
            if not df_processed[timestamp_col].isnull().all():
                log(f"    - Ordenando DataFrame por '{timestamp_col}'...")
                df_processed.sort_values(by=timestamp_col, inplace=True)
                df_processed.reset_index(drop=True, inplace=True)
            else:
                log(f"    - Coluna de Timestamp encontrada, mas vazia ou inválida. Não foi possível ordenar.")
                timestamp_col = None 
        else:
            log(f"    - Aviso: Coluna de Timestamp '{timestamp_col}' não selecionada ou não encontrada. O stream seguirá a ordem do CSV.")
            timestamp_col = None 

        log("[Passo 4/7] Convertendo valores Infinitos (inf) para NaN...")
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        log("[Passo 5/7] Removendo colunas, tratando nulos e codificando rótulos...")
        
        if target_label_col not in df_processed.columns:
            log(f"    - ERRO: Coluna de rótulo '{target_label_col}' não encontrada.")
            return None, None, None, None, log_messages, None
            
        le = LabelEncoder()
        y_data_series = le.fit_transform(df_processed[target_label_col].astype(str))
        log(f"    - LabelEncoder criado e ajustado. {len(le.classes_)} classes encontradas (ex: {le.classes_[:3]}...).")
        
        cols_para_remover_normalizadas = [col.strip() for col in cols_para_remover]
        todas_cols_para_remover = [target_label_col] + cols_para_remover_normalizadas
        if timestamp_col:
            todas_cols_para_remover.append(timestamp_col)
            
        cols_existentes_para_remover = [col for col in todas_cols_para_remover if col in df_processed.columns]
        
        X_data_df = df_processed.drop(columns=cols_existentes_para_remover, errors='ignore')
        log(f"    - {len(cols_existentes_para_remover)} colunas removidas do conjunto de features (Ex: {cols_existentes_para_remover[:3]}...).")

        X_data_df_numeric = X_data_df.select_dtypes(include=np.number)
        non_numeric_cols = X_data_df.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_cols:
            log(f"    - Aviso: Removendo {len(non_numeric_cols)} colunas não numéricas que sobraram (ex: {non_numeric_cols[:3]}).")
        
        nan_counts = X_data_df_numeric.isnull().sum().sum()
        y_data_pd = pd.Series(y_data_series, index=X_data_df_numeric.index) 
        
        if nan_counts > 0:
            log(f"    - Imputando {nan_counts} valores nulos/infinitos com o método: '{imputation_method}'...")
            if imputation_method == 'Mediana':
                X_data_df_cleaned = X_data_df_numeric.fillna(X_data_df_numeric.median()).fillna(0)
            elif imputation_method == 'Média':
                X_data_df_cleaned = X_data_df_numeric.fillna(X_data_df_numeric.mean()).fillna(0)
            elif imputation_method == 'Preencher com 0':
                X_data_df_cleaned = X_data_df_numeric.fillna(0)
            else: 
                log(f"    - Removendo {nan_counts} linhas com valores nulos...")
                X_data_df_cleaned = X_data_df_numeric.dropna()
                y_data_pd = y_data_pd.loc[X_data_df_cleaned.index]
        else:
            log("    - Nenhum valor nulo/infinito encontrado.")
            X_data_df_cleaned = X_data_df_numeric
        
        X_data_df_cleaned = X_data_df_cleaned.reset_index(drop=True)
        y_data_final = y_data_pd.reset_index(drop=True).values
        
        log(f"[Passo 6/7] Executando Método de Seleção de Features: '{feature_selection_method}'...")
        
        if feature_selection_method == 'Seleção Manual':
            if manual_features_list:
                log(f"    - Aplicando seleção manual. Mantendo {len(manual_features_list)} colunas.")
                features_selecionadas_clean = [col.strip() for col in manual_features_list]
                features_existentes = [col for col in features_selecionadas_clean if col in X_data_df_cleaned.columns]
                features_faltantes = set(features_selecionadas_clean) - set(features_existentes)
                
                if features_faltantes:
                    log(f"    - Aviso: As seguintes features não foram encontradas e serão ignoradas: {features_faltantes}")
                
                if not features_existentes:
                    log("    - ERRO: Nenhuma das features selecionadas foi encontrada no DataFrame. Abortando.")
                    return None, None, None, None, log_messages, None
                    
                X_data_df_cleaned = X_data_df_cleaned[features_existentes]
            else:
                log("    - Seleção Manual escolhida, mas nenhuma feature foi selecionada. Usando todas as features restantes.")
                feature_report['method'] = 'Manual (All)'

        elif feature_selection_method == 'Remover Correlação':
            log(f"    - Calculando matriz de correlação ({correlation_method})...")
            corr_matrix = X_data_df_cleaned.corr(method=correlation_method).abs()
            
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            
            log(f"    - Identificadas {len(to_drop)} features com correlação > {correlation_threshold}.")
            
            feature_report['dropped_features'] = to_drop
            feature_report['method'] = f'Correlation ({correlation_method})'
            
            if to_drop:
                log(f"    - Removendo features redundantes: {to_drop}")
                X_data_df_cleaned.drop(columns=to_drop, inplace=True)
            else:
                log("    - Nenhuma feature excedeu o limiar de correlação.")

        log("[Passo 7/7] Criando objeto NumpyStream...")
        X_data = X_data_df_cleaned.values.astype(np.float64)
        y_data = y_data_final
        
        log(f"    - Dados finais preparados: X_shape={X_data.shape}, y_shape={y_data.shape}.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            stream = NumpyStream(
                X_data,
                y_data,
                target_name=target_label_col, 
                dataset_name=file_path.split('/')[-1] 
            )
            
        stream.restart() 
        log("✅ Stream criado com sucesso e pronto para uso.")
        
        return stream, le, X_data_df_cleaned, df_processed, log_messages, feature_report
        
    except Exception as e:
        log(f"❌ ERRO INESPERADO NO PIPELINE: {e}")
        return None, None, None, None, log_messages, None