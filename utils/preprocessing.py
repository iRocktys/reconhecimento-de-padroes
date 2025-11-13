# utils/preprocessing.py

import pandas as pd

def perform_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função principal para aplicar o pré-processamento no dataset.
    
    Args:
        df: O DataFrame de entrada.
        
    Returns:
        O DataFrame processado.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # --- SIMULAÇÃO DE PRÉ-PROCESSAMENTO ---
    
    # 1. Tratamento de Colunas de Data (Exemplo)
    if 'date' in df.columns and not pd.api.types.is_datetime_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 2. Codificação (Exemplo: Label Encoding de 'season')
    if 'season' in df.columns:
        mapping = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        df['season_encoded'] = df['season'].map(mapping).fillna(df['season'].astype('category').cat.codes)
        
    # 3. Remoção de colunas desnecessárias para o modelo (Exemplo)
    if 'holiday' in df.columns:
        df = df.drop(columns=['holiday'])

    return df

# Você pode adicionar outras funções de utilidade aqui, como one-hot encoding, etc.