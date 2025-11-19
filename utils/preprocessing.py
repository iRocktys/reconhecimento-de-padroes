import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from capymoa.stream import NumpyStream

def create_stream_pipeline(
    file_path, 
    target_label_col, 
    timestamp_col, 
    cols_para_remover, 
    imputation_method='Mediana', 
    feature_selection_method='Seleção Manual', 
    n_features_auto=10, 
    manual_features_list=None,
    n_estimators=100, 
    rf_max_depth=None,
    rf_min_samples_leaf=1,
    rf_iterations=1,
    skb_score_func_name='f_classif',
    pca_svd_solver='auto',
    pca_whiten=False
):
    log_messages = []
    
    def log(message):
        log_messages.append(message)

    if NumpyStream is None:
        log("❌ ERRO CRÍTICO: A biblioteca 'capymoa' não foi encontrada. Instale-a com 'pip install capymoa'")
        return None, None, None, None, log_messages, None

    feature_importance_report = None

    try:
        log(f"--- Iniciando Pipeline: {file_path} ---")
        
        # --- Carregar Dados ---
        log("[Passo 1/7] Carregando arquivo CSV completo...")
        df = pd.read_csv(file_path)
        df_processed = df.copy()
        log(f"    - Arquivo carregado. Shape inicial: {df_processed.shape}")

        # --- Renomear Colunas ---
        log("[Passo 2/7] Limpando nomes das colunas (removendo espaços)...")
        df_processed.columns = df_processed.columns.str.strip()
        target_label_col = target_label_col.strip()
        if timestamp_col:
            timestamp_col = timestamp_col.strip()
        log("    - Colunas limpas.")

        # --- Ordenar por Timestamp ---
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

        # --- Tratar Infinitos ---
        log("[Passo 4/7] Convertendo valores Infinitos (inf) para NaN...")
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # --- Limpeza e Preparação de X/y ---
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
        original_features = X_data_df_cleaned.columns.tolist()
        
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

        elif feature_selection_method == 'Random Forest Importance':
            log(f"    - Iniciando {rf_iterations} iteração(ões) de RandomForest com {n_estimators} árvores cada...")
            all_importances = [] 
            
            for i in range(rf_iterations):
                log(f"      - Iteração {i+1}/{rf_iterations}...")
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=rf_max_depth,
                    min_samples_leaf=rf_min_samples_leaf,
                    random_state=42 + i, 
                    n_jobs=-1
                )
                rf.fit(X_data_df_cleaned, y_data_final)
                all_importances.append(rf.feature_importances_)
            
            avg_importances = pd.Series(np.mean(all_importances, axis=0), index=original_features)
            feature_importance_report = avg_importances.to_dict()
            
            top_features = avg_importances.nlargest(n_features_auto).index.tolist()
            log(f"    - Features selecionadas (baseado na média de {rf_iterations} rodadas): {top_features}")
            X_data_df_cleaned = X_data_df_cleaned[top_features]

        elif feature_selection_method == 'SelectKBest':
            if skb_score_func_name == 'f_classif':
                score_func = f_classif
                log_func_name = "ANOVA (f_classif)"
            elif skb_score_func_name == 'mutual_info_classif':
                score_func = mutual_info_classif
                log_func_name = "Informação Mútua"
            else:
                log(f"    - Aviso: Função de score '{skb_score_func_name}' desconhecida. Usando 'f_classif'.")
                score_func = f_classif
                log_func_name = "ANOVA (f_classif)"

            log(f"    - Aplicando SelectKBest (função: {log_func_name}) para encontrar as {n_features_auto} melhores features...")
            
            k = min(n_features_auto, len(original_features))
            
            selector = SelectKBest(score_func, k=k)
            X_new = selector.fit_transform(X_data_df_cleaned, y_data_final)
            
            scores = pd.Series(selector.scores_, index=original_features)
            feature_importance_report = scores.to_dict()

            top_features = selector.get_feature_names_out(original_features).tolist()
            log(f"    - Features selecionadas: {top_features}")
            X_data_df_cleaned = pd.DataFrame(X_new, columns=top_features)

        elif feature_selection_method == 'PCA (Extração de Componentes)':
            log(f"    - Aplicando PCA (solver: '{pca_svd_solver}', whiten: {pca_whiten}) para extrair {n_features_auto} componentes...")
            n_components = min(n_features_auto, len(original_features))
            
            pca = PCA(
                n_components=n_components,
                svd_solver=pca_svd_solver,
                whiten=pca_whiten,
                random_state=42
            )
            
            X_pca = pca.fit_transform(X_data_df_cleaned)
            
            pca_features = [f"PCA_{i+1}" for i in range(n_components)]
            log(f"    - Componentes extraídos: {pca_features}")
            X_data_df_cleaned = pd.DataFrame(X_pca, columns=pca_features)
            
            explained_variance = pca.explained_variance_ratio_
            feature_importance_report = {f"PCA_{i+1}": variance for i, variance in enumerate(explained_variance)}
            log(f"    - Variância explicada total: {sum(explained_variance)*100:.2f}%")

        # --- Criar Stream ---
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
        
        return stream, le, X_data_df_cleaned, df_processed, log_messages, feature_importance_report
        
    except Exception as e:
        log(f"❌ ERRO INESPERADO NO PIPELINE: {e}")
        return None, None, None, None, log_messages, None