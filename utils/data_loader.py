import os
import pandas as pd
import streamlit as st
import time

# --- Constantes ---
PANDAS_CHUNK_SIZE = 50000 
ATTACK_LABEL_COL = 'Label'
COLUMNS_TO_DROP = [
    'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
    'Destination IP', 'Destination Port', 'Timestamp', 
    'SimillarHTTP', 'Fwd Header Length.1', 'Protocol'
]
MIN_SAMPLES_PER_CHUNK = 1000
BENIGN_LABEL = 'BENIGN'

# Fatores de downsample padrão
DOWNSAMPLE_FACTORS = {
    'DrDoS_NTP': 0.01,
    'DrDoS_DNS': 0.01,
    'DrDoS_LDAP': 0.01,
    'DrDoS_MSSQL': 0.001,
    'DrDoS_NetBIOS': 0.01,
    'DrDoS_SNMP': 0.01,
    'DrDoS_SSDP': 0.01,
    'DrDoS_UDP': 0.01,
    'UDPLag': 0.01,
    'Syn': 0.001,
    'TFTP': 0.001,
    'Portmap': 0.01,
    'NetBIOS': 0.01,
    'LDAP': 0.01,
    'MSSQL': 0.001,
    'UDP': 0.01,
    'Default': 0.01
}

# --- CORREÇÃO DE LÓGICA ---
# Voltando ao seu rascunho original, que estava correto.
# Os nomes dos arquivos .csv estão na lista.
ATTACK_ORDER = {
    '03-11': [
        'Portmap.csv', 'NetBIOS.csv', 'LDAP.csv', 'MSSQL.csv', 'UDP.csv', 'UDPLag.csv', 'Syn.csv'
    ],
    '01-12': [
        'DrDoS_NTP.csv', 'DrDoS_DNS.csv', 'DrDoS_LDAP.csv', 'DrDoS_MSSQL.csv', 
        'DrDoS_NetBIOS.csv', 'DrDoS_SNMP.csv', 'DrDoS_SSDP.csv', 'DrDoS_UDP.csv', 
        'UDPLag.csv', 'Syn.csv', 'TFTP.csv'
    ]
}

# Arquivos de saída
OUTPUT_FILES = {
    '03-11': 'CICDDoS2019_03_11.csv',
    '01-12': 'CICDDoS2019_01_12.csv'
}

# --- Funções de Lógica ---

def processar_e_salvar_dia(
    dia, 
    dataset_path, 
    dynamic_downsample_factors,
    progress_placeholder,
    cancel_flag_getter
):
    """
    Processa todos os arquivos CSV de um dia, aplica downsampling dinâmico
    e salva em um único arquivo de saída.
    Agora verifica a flag de cancelamento.
    """
    
    lista_arquivos = ATTACK_ORDER[dia]
    output_filepath = OUTPUT_FILES[dia]
    total_amostras_mantidas = 0
    header_escrito = False
    
    status_text = progress_placeholder.empty()

    with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
        
        for i, filename in enumerate(lista_arquivos):
            filepath = os.path.join(dataset_path, dia, filename)
            attack_name_from_file = filename.replace('.csv', '')
            
            status_text.info(f"Procurando por: {filepath} ({i+1}/{len(lista_arquivos)})...")
            
            if not os.path.exists(filepath):
                status_text.warning(f"Atenção: Arquivo não encontrado, pulando: {filepath}")
                time.sleep(1)
                continue
                
            try:
                csv_reader = pd.read_csv(
                    filepath, 
                    chunksize=PANDAS_CHUNK_SIZE, 
                    low_memory=False, 
                    on_bad_lines='skip',
                    encoding='utf-8',
                    engine='c'
                )
            except Exception as e:
                status_text.error(f"Erro ao ler {filename}: {e}. Pulando...")
                time.sleep(2)
                continue
                
            for df_chunk in csv_reader:
                
                # --- ADIÇÃO: Verifica a flag de cancelamento ---
                if not cancel_flag_getter():
                    status_text.warning("Cancelamento solicitado. Parando o processamento...")
                    # Retorna o que foi feito até agora e um status de "Cancelado"
                    return total_amostras_mantidas, output_filepath, "Cancelled"
                
                df_chunk.columns = df_chunk.columns.str.strip()
                cols_existentes_drop = [col for col in COLUMNS_TO_DROP if col in df_chunk.columns]
                df_chunk = df_chunk.drop(columns=cols_existentes_drop, errors='ignore')
                
                if ATTACK_LABEL_COL not in df_chunk.columns:
                    status_text.warning(f"Coluna '{ATTACK_LABEL_COL}' não encontrada no chunk de {filename}. Pulando chunk.")
                    continue
                
                df_chunk[ATTACK_LABEL_COL] = df_chunk[ATTACK_LABEL_COL].apply(
                    lambda x: BENIGN_LABEL if 'BENIGN' in str(x).upper() else attack_name_from_file
                )
                
                df_benign = df_chunk[df_chunk[ATTACK_LABEL_COL] == BENIGN_LABEL]
                df_ataque = df_chunk[df_chunk[ATTACK_LABEL_COL] != BENIGN_LABEL]

                df_ataque_downsampled = df_ataque

                if not df_ataque.empty:
                    factor = dynamic_downsample_factors.get(attack_name_from_file, DOWNSAMPLE_FACTORS['Default'])
                    
                    if len(df_ataque) < MIN_SAMPLES_PER_CHUNK:
                        factor = 1.0
                    
                    if factor < 1.0:
                        df_ataque_downsampled = df_ataque.sample(
                            frac=factor, 
                            random_state=42
                        )

                df_chunk_reduzido = pd.concat([df_benign, df_ataque_downsampled]).sample(frac=1, random_state=42)
                
                if df_chunk_reduzido.empty:
                    continue

                df_chunk_reduzido.to_csv(
                    f, 
                    index=False, 
                    header=not header_escrito, 
                    mode='a' 
                )
                header_escrito = True
                total_amostras_mantidas += len(df_chunk_reduzido)

    status_text.empty()
    
    if total_amostras_mantidas == 0:
        status_text.error("Processamento concluído, mas 0 amostras foram salvas. Verifique se o caminho no Passo 1 está correto e se as subpastas (03-11, 01-12) contêm os arquivos .csv.")
    
    # Retorna o total, o caminho e um status de "Sucesso"
    return total_amostras_mantidas, output_filepath, "Success"

@st.cache_data
def get_processed_file_report(filepath):
    """
    Lê o arquivo CSV processado e retorna um dataframe com a contagem de labels.
    Cacheado para ser rápido.
    """
    if not os.path.exists(filepath):
        # Se o arquivo não existe, não é um erro, só não foi criado.
        return None
        
    try:
        # Lê apenas a coluna necessária para o relatório
        df = pd.read_csv(filepath, usecols=[ATTACK_LABEL_COL])
        report = df[ATTACK_LABEL_COL].value_counts().reset_index()
        report.columns = ['Label', 'Contagem']
        return report
    
    except pd.errors.EmptyDataError:
        # O arquivo foi criado, mas está vazio (0 amostras). Retorna um DF vazio.
        st.warning("O arquivo de relatório está vazio (provavelmente 0 amostras foram processadas).")
        return pd.DataFrame(columns=["Label", "Contagem"])
        
    except Exception as e:
        # Um erro real de leitura
        st.error(f"Erro ao ler o relatório do arquivo: {e}")
        return None