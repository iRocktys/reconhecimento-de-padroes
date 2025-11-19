import streamlit as st
import os
import pandas as pd
import altair as alt 
from utils.data_loader import (
    ATTACK_ORDER, 
    DOWNSAMPLE_FACTORS, 
    process_and_save,
    get_processed_file_report,
    list_data_files, 
    BENIGN_LABEL,
    DATA_DIR 
)
from utils.style import load_custom_css
load_custom_css("style.css")

st.set_page_config(
    page_title="IDS Stream Mining", 
    page_icon="🛡️",
    layout="centered" 
)

# Gerenciamento de Estado
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_filepath' not in st.session_state:
    st.session_state.processed_filepath = None
if 'processed_amostras' not in st.session_state:
    st.session_state.processed_amostras = 0
if 'file_to_analyze' not in st.session_state:
    st.session_state.file_to_analyze = None 

# Funções de Callback
def start_processing():
    st.session_state.processing = True
    st.session_state.processed_filepath = None
    st.session_state.processed_amostras = 0

def cancel_processing():
    st.session_state.processing = False

def get_state():
    return st.session_state.processing

# Funções da Página
def render_sliders(selected_day):
    attack_files = ATTACK_ORDER[selected_day]
    attack_names = [f.replace('.csv', '') for f in attack_files]
    
    st.markdown("""
    Use os seletores abaixo para definir a fração (porcentagem) de cada ataque que será mantida.
    * **O que é isso?** Estamos fazendo um *Downsampling* (redução de amostras).
    * **Exemplo:** Um valor `0.01` significa "manter apenas 1% das amostras desse ataque". Um valor `1.0` significa "manter 100%".
    """)
    
    st.warning(f"⚠️ **{BENIGN_LABEL}**: Mantido em **1.0 (100%)**. Esta opção é travada, pois o tráfego normal ('BENIGN') é raro e muito importante.")
    st.markdown("---")
    
    dynamic_factors = {}
    col1, col2 = st.columns(2)
    
    for i, attack_name in enumerate(attack_names):
        default_factor = DOWNSAMPLE_FACTORS.get(attack_name, DOWNSAMPLE_FACTORS['Default'])
        target_col = col1 if i % 2 == 0 else col2
        
        slider_value = target_col.slider(
            f"**{attack_name}**",
            min_value=0.001,
            max_value=1.0,
            value=default_factor,
            step=0.001,
            format="%.3f"
        )
        dynamic_factors[attack_name] = slider_value
        
    return dynamic_factors

def display_report():
    filepath = st.session_state.get('file_to_analyze')
    
    if not filepath:
        st.info("Nenhum arquivo selecionado. Processe, selecione ou faça upload de um arquivo na seção **'Seleção dos Dados'**.")
        return
        
    if not os.path.exists(filepath):
        st.error(f"Arquivo selecionado '{filepath}' não foi encontrado.")
        st.session_state.file_to_analyze = None 
        return

    report_df = get_processed_file_report(filepath)
    
    if report_df is None:
        st.error(f"Falha ao ler o arquivo de relatório '{filepath}'.")
    elif report_df.empty:
        st.warning(f"O arquivo selecionado '{filepath}' está vazio.")
    else:
        st.success(f"Exibindo análise para: **{filepath}**")
        st.subheader("Visualização da Distribuição")
        
        chart = alt.Chart(report_df).mark_bar().encode(
            x=alt.X('Label', sort=None),
            y=alt.Y('Contagem'), 
            color=alt.Color('Label', legend=alt.Legend(
                title="Legenda", 
                orient='right'
            )),
            tooltip=['Label', 'Contagem'] 
        ).interactive() 
        
        st.altair_chart(chart, width='stretch')
        st.subheader("Contagem de Amostras")
        st.dataframe(
            report_df,
            hide_index=True,
            column_config={
                "Contagem": st.column_config.NumberColumn(format="%d")
            },
            width='stretch'
        )
        
        st.info(f"**Próximo Passo:** O arquivo {filepath} está pronto. Clique em **'Pré-processamento'** na barra lateral para continuar.")

st.title("Base de Dados e Carregamento")

with st.container(border=True):
    st.header("Localizar o Dataset", divider="rainbow")
    st.markdown("Insira o **caminho completo** para a pasta principal `CICDDoS2019/`. O aplicativo irá procurar as subpastas (`01-12`, `03-11`) dentro desse caminho.")
    dataset_path = st.text_input(
        "Insira o caminho para a pasta:", 
        "C:/GitHub/anomaly-detection-data-stream/datasets/CICDDoS2019",
        placeholder="Ex: C:/Users/SeuUser/Desktop/datasets/CICDDoS2019/"
    )
    path_exists = os.path.exists(dataset_path)
    if not path_exists:
        st.error(f"Caminho não encontrado: '{dataset_path}'.")
    else:
        st.success(f"Caminho encontrado: '{dataset_path}'")

    selected_day = st.selectbox(
        "Escolha qual dos dias (subpastas) você deseja processar:",
        options=list(ATTACK_ORDER.keys()),
        key="selected_day"
    )

with st.container(border=True):
    st.header("Configurar o Downsampling", divider="rainbow")
    st.markdown("O dataset é **extremamente desbalanceado**. Use os controles abaixo para reduzir o número de amostras das classes de ataque mais comuns.")
    with st.expander("Clique para configurar o downsampling", expanded=True):
        st.info("""
        **Por que fazer isso?**
        Manter 50 milhões de amostras do ataque 'MSSQL' não ensina nada de novo ao modelo e torna o treinamento lento. É melhor manter 50.000 amostras (0.1%) de 'MSSQL'.
        """)
        dynamic_factors = render_sliders(selected_day)

with st.container(border=True):
    st.header("Processar e Salvar o Arquivo", divider="rainbow")
    st.markdown("Defina o nome do arquivo de saída (ele será salvo na pasta `data/`) e inicie o processo.")

    default_filename = f"CICDDoS2019_{selected_day}_processado.csv"
    output_filename = st.text_input(
        "Nome do arquivo de saída:",
        value=default_filename,
        help="O arquivo será salvo no diretório 'data/' do seu projeto."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Iniciar Processamento", 
            on_click=start_processing,
            disabled=st.session_state.processing or not path_exists or not output_filename,
            type="primary"
        ):
            if not output_filename.endswith('.csv'):
                st.error("O nome do arquivo deve terminar com '.csv'")
                st.session_state.processing = False
                st.rerun()

if st.session_state.processing:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("❌ Cancelar Processamento", on_click=cancel_processing)
    
    progress_placeholder = st.empty()
    
    with st.spinner("Executando... Isso pode levar vários minutos."):
        try:
            total_amostras, output_filepath, status = process_and_save(
                dia=selected_day,
                dataset_path=dataset_path,
                dynamic_downsample_factors=dynamic_factors,
                output_filename=output_filename, 
                progress_placeholder=progress_placeholder,
                cancel_flag_getter=get_state
            )
            
            if status == "Success":
                st.success(f"**Sucesso!** O arquivo processado **'{output_filepath}'** foi salvo com **{total_amostras:,}** amostras.")
                st.session_state.processed_filepath = output_filepath
                st.session_state.processed_amostras = total_amostras
                st.session_state.file_to_analyze = output_filepath
            
            elif status == "Cancelled":
                st.warning("Operação cancelada pelo usuário.")
                if os.path.exists(output_filepath):
                    os.remove(output_filepath)
                st.session_state.processed_filepath = None

        except Exception as e:
            st.error(f"Ocorreu um erro crítico durante o processamento: {e}")
            st.exception(e)
        
        st.session_state.processing = False
        st.rerun() 

with st.container(border=True):
    st.header("Seleção dos Dados", divider="rainbow")
    st.markdown("Selecione o arquivo que deseja analisar. Você pode usar o resultado do processamento, escolher um arquivo `.csv` já existente na pasta `data/`, ou fazer o upload de um novo arquivo.")

    tab1, tab2, tab3 = st.tabs([
        "🎯 Resultado do Processamento", 
        "📂 Selecionar", 
        "⬆️ Fazer Upload"
    ])

    with tab1:
        st.markdown("Esta opção analisa o último arquivo que foi gerado com sucesso no ``Processar e Salvar o Arquivo``.")
        if st.session_state.processed_filepath:
            st.info(f"Arquivo do processamento: `{st.session_state.processed_filepath}`")
            if st.button("Analisar este arquivo"):
                st.session_state.file_to_analyze = st.session_state.processed_filepath
                st.rerun()
        else:
            st.warning("Nenhum arquivo foi processado nesta sessão ainda.")

    with tab2:
        st.markdown(f"Estes são os arquivos `.csv` encontrados na sua pasta `{DATA_DIR}/`.")
        data_files = list_data_files()
        
        if not data_files:
            st.info(f"Nenhum arquivo `.csv` encontrado na pasta `{DATA_DIR}/`.")
        else:
            selected_file = st.selectbox("Escolha um arquivo existente:", options=data_files)
            if selected_file:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Analisar arquivo selecionado"):
                        st.session_state.file_to_analyze = os.path.join(DATA_DIR, selected_file)
                        st.rerun()
    with tab3:
        st.markdown(f"Faça o upload de um arquivo `.csv`. Ele será salvo na pasta `{DATA_DIR}/` e selecionado para análise.")
        uploaded_file = st.file_uploader("Escolha um arquivo .csv", type="csv")
        
        if uploaded_file is not None:
            save_path = os.path.join(DATA_DIR, uploaded_file.name)
            
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(f"Analisar arquivo '{uploaded_file.name}'"):
                        st.session_state.file_to_analyze = save_path
                        st.rerun()
                    
            except Exception as e:
                st.error(f"Erro ao salvar o arquivo: {e}")

with st.container(border=True):
    st.header("Análise do Arquivo Selecionado", divider="rainbow")
    display_report()