# pages/2_Base_de_Dados.py
import streamlit as st
import pandas as pd
import os
import altair as alt

st.title("📊 Seleção e Análise da Base de Dados")
st.write("Escolha um arquivo CSV da pasta `/data` e visualize seus dados.")

# Define as colunas a serem usadas nos gráficos (agora são minúsculas e sem espaço)
LABEL_COLUMN = 'label'      
TIMESTAMP_COLUMN = 'timestamp' 

# Caminho e verificação da pasta
DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    st.error(f"A pasta '{DATA_FOLDER}' não foi encontrada.")
    st.stop()

csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
if not csv_files:
    st.warning(f"Nenhum arquivo CSV encontrado na pasta `{DATA_FOLDER}`.")
    st.stop()


# --- 1. SELEÇÃO DO ARQUIVO ---
# BLINDAGEM: Usa .get() para acessar o estado, garantindo um valor padrão se a chave não existir.
df_original = st.session_state.get('df_original')
selected_csv_name_state = st.session_state.get('selected_csv_name')

default_index = 0
if selected_csv_name_state in csv_files:
    default_index = csv_files.index(selected_csv_name_state)

selected_csv = st.selectbox(
    "Selecione um arquivo CSV para carregar",
    csv_files,
    index=default_index,
    key='csv_selector'
)

# --- 2. LÓGICA DE CARREGAMENTO E LIMPEZA DE COLUNAS ---
if selected_csv:
    file_path = os.path.join(DATA_FOLDER, selected_csv)
    
    # BLINDAGEM: Usa df_original e selected_csv_name_state (já obtidos com .get())
    is_same_file = (df_original is not None) and (selected_csv_name_state == selected_csv)

    if not is_same_file:
        with st.spinner(f"Carregando e limpando colunas de {selected_csv}..."):
            try:
                df = pd.read_csv(file_path)
                
                # --- LIMPEZA DE COLUNAS ---
                df.columns = [col.strip().lower().replace(' ', '_').replace('/', '_') for col in df.columns]
                
                st.session_state.df_original = df.copy() # Salva o DF
                st.session_state.selected_csv_name = selected_csv # Salva o nome
                st.session_state.df_processed = None # Reseta o processado
                st.success(f"Arquivo **{selected_csv}** carregado! ({df.shape[0]} linhas, {df.shape[1]} colunas)")
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo CSV '{selected_csv}': {e}")
                st.session_state.df_original = None
                st.session_state.selected_csv_name = None
                st.stop()
    else:
        df = df_original.copy()
        st.info(f"Arquivo **{selected_csv}** já está carregado em memória.")

    # --- 3. VISUALIZAÇÃO ---
    
    st.subheader("Visualização dos Dados (Primeiras 5 Linhas)")
    st.dataframe(df.head(), width='stretch') 

    st.markdown("---")
    st.subheader("Gráficos de Análise Exploratória")

    # --- 1. Gráfico de Quantidade de Rótulos ---
    if LABEL_COLUMN in df.columns:
        st.write(f"### 🎯 Distribuição da Coluna '{LABEL_COLUMN}'")
        label_counts = df[LABEL_COLUMN].value_counts().reset_index()
        label_counts.columns = [LABEL_COLUMN, 'Count']

        chart_labels = alt.Chart(label_counts).mark_bar().encode(
            x=alt.X('Count', title='Quantidade'),
            y=alt.Y(LABEL_COLUMN, sort='-x', title='Rótulo'),
            tooltip=[LABEL_COLUMN, 'Count']
        ).properties(
            title=f'Quantidade de Rótulos ({LABEL_COLUMN})'
        )
        st.altair_chart(chart_labels, width='stretch') 
    else:
        st.warning(f"Coluna de rótulo '{LABEL_COLUMN}' não encontrada.")

    # --- 2. Gráfico de Rótulos Distribuídos ao Longo do Tempo ---
    if TIMESTAMP_COLUMN in df.columns and LABEL_COLUMN in df.columns:
        st.write(f"### 📊 Rótulos Distribuídos ao Longo do Tempo")
        
        try:
            df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors='coerce')
            df_cleaned = df.dropna(subset=[TIMESTAMP_COLUMN, LABEL_COLUMN])
            
            if not df_cleaned.empty:
                df_time_labels = df_cleaned.groupby([
                    pd.Grouper(key=TIMESTAMP_COLUMN, freq='1min'), 
                    LABEL_COLUMN
                ]).size().reset_index(name='Count')

                chart_flow = alt.Chart(df_time_labels).mark_area(opacity=0.7).encode(
                    x=alt.X(TIMESTAMP_COLUMN, title='Tempo (Agregado por Minuto)'),
                    y=alt.Y('Count', title='Contagem de Ocorrências'),
                    color=alt.Color(LABEL_COLUMN, title='Rótulo'),
                    tooltip=[TIMESTAMP_COLUMN, LABEL_COLUMN, 'Count']
                ).properties(
                    title=f'Contagem de Rótulos ({LABEL_COLUMN}) ao Longo do Tempo'
                )
                st.altair_chart(chart_flow, width='stretch')
            else:
                 st.warning("Não há dados válidos de tempo ou rótulo após a limpeza para o gráfico de distribuição temporal.")

        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico de distribuição de rótulos ao longo do tempo. Erro: {e}")
    else:
        st.warning(f"Colunas '{TIMESTAMP_COLUMN}' e/ou '{LABEL_COLUMN}' não encontradas no dataset para o gráfico de distribuição temporal.")