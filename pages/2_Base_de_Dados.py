# pages/2_Base_de_Dados.py
import streamlit as st
import pandas as pd
import os
import altair as alt

st.title("Seleção e Análise da Base de Dados")
st.write("Escolha um arquivo CSV da pasta `/data` e visualize seus dados.")

DATA_FOLDER = "data"

csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

if not csv_files:
    st.warning(f"Nenhum arquivo CSV encontrado na pasta `{DATA_FOLDER}`.")
    st.stop()

# --- SELEÇÃO DO ARQUIVO  ---
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

# --- LÓGICA DE CARREGAMENTO E LIMPEZA DE COLUNAS ---
if selected_csv:
    file_path = os.path.join(DATA_FOLDER, selected_csv)
    
    is_same_file = (st.session_state.df_original is not None) and (st.session_state.selected_csv_name == selected_csv)

    if not is_same_file:
        with st.spinner(f"Carregando e limpando colunas de {selected_csv}..."):
            try:
                df = pd.read_csv(file_path)
                
                # --- CORREÇÃO DE LIMPEZA DE COLUNAS ---
                df.columns = [col.strip().lower() for col in df.columns]
                
                st.session_state.df_original = df.copy()
                st.session_state.selected_csv_name = selected_csv
                st.session_state.df_processed = None
                st.success(f"Arquivo **{selected_csv}** carregado! ({df.shape[0]} linhas, {df.shape[1]} colunas)")
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo CSV '{selected_csv}': {e}")
                st.session_state.df_original = None
                st.session_state.selected_csv_name = None
                st.stop()
    else:
        df = st.session_state.df_original.copy()
        st.info(f"Arquivo **{selected_csv}** já está carregado em memória.")

    # --- VISUALIZAÇÃO ---
    st.subheader("Visualização dos Dados Carregados")
    st.dataframe(df.head(), width='stretch') 

    st.markdown("---")
    st.title("Gráficos de Análise Exploratória")

    # Define as colunas a serem usadas nos gráficos 
    label_column = 'label'      
    timestamp_column = 'timestamp' 

    # --- Gráfico de Quantidade de Rótulos ---
    if label_column in df.columns:
        label_counts = df[label_column].value_counts().reset_index()
        label_counts.columns = [label_column, 'Count']

        chart_labels = alt.Chart(label_counts).mark_bar().encode(
            x=alt.X('Count', title='Quantidade'),
            y=alt.Y(label_column, sort='-x', title='Rótulo'),
            tooltip=[label_column, 'Count']
        ).properties(
            title=f'Quantidade de Rótulos ({label_column})'
        )
        st.altair_chart(chart_labels, width='stretch') 
    else:
        st.warning("Coluna de rótulo não encontrada para o gráfico de quantidade de rótulos.")

    # --- Gráfico de Rótulos Distribuídos ao Longo do Tempo ---
    if timestamp_column in df.columns and label_column in df.columns:        
        try:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
            
            df_cleaned = df.dropna(subset=[timestamp_column, label_column])
            
            if not df_cleaned.empty:
                df_time_labels = df_cleaned.groupby([
                    pd.Grouper(key=timestamp_column, freq='1min'), 
                    label_column
                ]).size().reset_index(name='Count')

                chart_flow = alt.Chart(df_time_labels).mark_area(opacity=0.7).encode(
                    x=alt.X(timestamp_column, title='Tempo (Agregado por Minuto)'),
                    y=alt.Y('Count', title='Contagem de Ocorrências'),
                    color=alt.Color(label_column, title='Rótulo'),
                    tooltip=[timestamp_column, label_column, 'Count']
                ).properties(
                    title=f'Contagem de Rótulos ({label_column}) ao Longo do Tempo'
                )
                st.altair_chart(chart_flow, width='stretch')
            else:
                 st.warning("Não há dados válidos de tempo ou rótulo após a limpeza para o gráfico de distribuição temporal.")

        except Exception as e:
            st.warning(f"Não foi possível gerar o gráfico de distribuição de rótulos ao longo do tempo. Erro: {e}")
    else:
        st.warning(f"Colunas '{timestamp_column}' e/ou '{label_column}' não encontradas no dataset para o gráfico de distribuição temporal.")