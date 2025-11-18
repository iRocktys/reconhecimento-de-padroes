import streamlit as st
import os
import pandas as pd
from utils.style import load_custom_css
load_custom_css("style.css")

# Este import é essencial e busca no outro arquivo
from utils.data_loader import (
    ATTACK_ORDER, 
    DOWNSAMPLE_FACTORS, 
    processar_e_salvar_dia,
    get_processed_file_report,
    BENIGN_LABEL
)

st.set_page_config(
    page_title="Carregamento de Dados", 
    page_icon="📦",
    # --- MUDANÇA AQUI ---
    # Mudei de "wide" para "centered" para centralizar
    # a página e diminuir a largura total.
    layout="centered" 
)

# --- Gerenciamento de Estado ---
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_filepath' not in st.session_state:
    st.session_state.processed_filepath = None
if 'processed_amostras' not in st.session_state:
    st.session_state.processed_amostras = 0

def start_processing():
    """Callback para iniciar o processamento."""
    st.session_state.processing = True
    st.session_state.processed_filepath = None
    st.session_state.processed_amostras = 0

def cancel_processing():
    """Callback para solicitar o cancelamento."""
    st.session_state.processing = False

def get_state():
    """Função auxiliar para passar o estado para a lógica."""
    return st.session_state.processing

# --- Funções da Página ---

def render_sliders(selected_day):
    """Cria dinamicamente os sliders para o dia selecionado."""
    
    attack_files = ATTACK_ORDER[selected_day]
    attack_names = [f.replace('.csv', '') for f in attack_files]
    
    st.markdown("""
    Use os seletores abaixo para definir a fração (porcentagem) de cada ataque que será mantida.
    
    * **O que é isso?** Estamos fazendo um *Downsampling* (redução de amostras).
    * **Exemplo:** Um valor `0.01` significa "manter apenas 1% das amostras desse ataque". Um valor `1.0` significa "manter 100%".
    """)
    
    st.warning(f"⚠️ **{BENIGN_LABEL}**: Mantido em **1.0 (100%)**. Esta opção é travada, pois o tráfego normal ('BENIGN') é raro e muito importante para o modelo aprender a diferenciar.")
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
    """
    Exibe o relatório do arquivo processado.
    Agora lê o estado do st.session_state.
    """
    
    filepath = st.session_state.processed_filepath
    
    if not filepath:
        st.info("Nenhum arquivo foi processado ainda. Assim que você processar um dia no Passo 4, os resultados aparecerão aqui.")
        return
        
    if not os.path.exists(filepath):
        st.error(f"O arquivo {filepath} não foi encontrado. Pode ter sido movido ou excluído.")
        return

    report_df = get_processed_file_report(filepath)
    
    if report_df is None:
        st.error("Falha ao ler o arquivo de relatório (pode estar corrompido).")
    elif report_df.empty:
        st.warning(f"O arquivo processado está vazio (Total de Amostras: {st.session_state.processed_amostras}). Verifique os logs e o Passo 1.")
    else:
        st.success(f"Análise do arquivo: **{filepath}** (Total de Amostras: {st.session_state.processed_amostras:,})")
        
        # O layout "centered" já limita a largura, então
        # os gráficos e tabelas não vão estourar.
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Contagem de Amostras")
            st.dataframe(
                report_df,
                # --- MUDANÇA AQUI ---
                # Removido 'width='stretch''
                hide_index=True,
                column_config={
                    "Contagem": st.column_config.NumberColumn(format="%d")
                }
            )
        with col2:
            st.subheader("Visualização da Distribuição")
            st.bar_chart(
                report_df,
                x='Label',
                y='Contagem',
                color='Label'
            )
        st.info(f"💾 **Próximo Passo:** O seu novo dataset está pronto.\n\nClique em **'2. Pré-processamento'** na barra lateral para continuar.")


# --- Renderização da Página ---

st.title("📦 1. Base de Dados e Carregamento")
st.markdown("""
Bem-vindo à primeira etapa! O objetivo aqui é carregar o imenso dataset **CICDDoS2019**, que está dividido em múltiplos arquivos, e prepará-lo para a análise.

Como este dataset é muito grande e desbalanceado (muitos ataques, pouco tráfego normal), vamos:
1.  Juntar todos os arquivos de um dia específico em um só.
2.  Reduzir de forma inteligente a quantidade de amostras de ataque (*downsampling*).
3.  Salvar um **novo arquivo CSV**, menor e mais balanceado, para usarmos nas próximas etapas.
""")

# --- Passo 1: Localizar o Dataset ---
st.header("Passo 1: Localizar o Dataset", divider="rainbow")
st.markdown("""
Como este aplicativo roda no servidor, ele não pode "adivinhar" onde os arquivos do dataset estão no computador.

Por favor, insira o **caminho completo** para a pasta principal `CICDDoS2019/`. O aplicativo irá então procurar as subpastas (`01-12`, `03-11`) dentro desse caminho.
""")

dataset_path = st.text_input(
    "Insira o caminho para a pasta 'datasets/CICDDoS2019/'", 
    "datasets/CICDDoS2019/",
    placeholder="Ex: C:/Users/SeuUser/Desktop/datasets/CICDDoS2019/"
)
path_exists = os.path.exists(dataset_path)
if not path_exists:
    st.error(f"Caminho não encontrado: '{dataset_path}'. Verifique o diretório.")
else:
    st.success(f"Caminho encontrado: '{dataset_path}'")

# --- Passo 2: Selecionar o Dia para Processar ---
st.header("Passo 2: Selecionar o Dia para Processar", divider="rainbow")
st.markdown("""
O dataset original foi capturado em dois dias diferentes, cada um com um conjunto diferente de ataques.
Escolha qual dos dias você deseja processar agora.
""")

selected_day = st.selectbox(
    "Escolha o conjunto de dados (dia):",
    options=list(ATTACK_ORDER.keys()),
    key="selected_day",
    help="O aplicativo irá procurar os arquivos CSV dentro da subpasta correspondente (ex: .../CICDDoS2019/01-12/)"
)

# --- Passo 3: Configurar o Downsampling ---
st.header("Passo 3: Configurar o Downsampling (Redução de Amostras)", divider="rainbow")
st.markdown("""
Esta é a etapa mais importante. O dataset é **extremamente desbalanceado**: algumas classes de ataque têm milhões de amostras, enquanto o tráfego 'BENIGN' (normal) é raro.
Se treinarmos um modelo com esses dados, ele ficará "viciado" em prever só o ataque mais comum.
""")

with st.expander("Clique para configurar o downsampling de cada classe de ataque", expanded=True):
    st.info("""
    **Por que fazer isso?**
    Manter 50 milhões de amostras do ataque 'MSSQL' não ensina nada de novo ao modelo e torna o treinamento impossivelmente lento. É melhor manter 50.000 amostras (0.1%) de 'MSSQL' e 100% das amostras 'BENIGN'.
    
    Isso torna o dataset **menor, mais rápido e mais balanceado**, o que resulta em um modelo final muito melhor.
    """)
    
    dynamic_factors = render_sliders(selected_day)

# --- Passo 4: Processar e Salvar ---
st.header("Passo 4: Processar e Salvar o Arquivo", divider="rainbow")
st.markdown("""
Pronto! Ao clicar no botão abaixo, o aplicativo irá:
1.  Ler, pedaço por pedaço (*chunk*), todos os arquivos CSV do dia selecionado.
2.  Aplicar as regras de *downsampling* que você definiu no Passo 3.
3.  Juntar tudo e salvar em um **único arquivo CSV** novo (ex: `CICDDoS2019_01_12.csv`).
""")

# --- MUDANÇA AQUI ---
# Removido 'width='stretch'' para o botão
# ficar do tamanho do texto.
col1, col2, col3 = st.columns([1, 2, 1])

with col2: # Colocamos o botão na coluna do meio
    st.button(
        "🚀 Iniciar Processamento e Concatenação", 
        on_click=start_processing,
        disabled=st.session_state.processing or not path_exists,
        type="primary"
    )

if st.session_state.processing:
    # --- MUDANÇA AQUI ---
    # Removido 'width='stretch''
    st.button(
        "❌ Cancelar Processamento", 
        on_click=cancel_processing
    )
    
    progress_placeholder = st.empty()
    
    with st.spinner("Executando... Isso pode levar vários minutos."):
        try:
            total_amostras, output_filepath, status = processar_e_salvar_dia(
                dia=selected_day,
                dataset_path=dataset_path,
                dynamic_downsample_factors=dynamic_factors,
                progress_placeholder=progress_placeholder,
                cancel_flag_getter=get_state # Passa a função que lê o state
            )
            
            if status == "Success":
                st.success(f"🎉 **Sucesso!** O arquivo processado **'{output_filepath}'** foi salvo com **{total_amostras:,}** amostras.")
                st.session_state.processed_filepath = output_filepath
                st.session_state.processed_amostras = total_amostras
            
            elif status == "Cancelled":
                st.warning("Operação cancelada pelo usuário.")
                if os.path.exists(output_filepath):
                    os.remove(output_filepath) # Limpa o arquivo incompleto
                st.session_state.processed_filepath = None

        except Exception as e:
            st.error(f"Ocorreu um erro crítico durante o processamento: {e}")
            st.exception(e) # Mostra o traceback completo do erro
        
        st.session_state.processing = False
        st.rerun() 

# --- Passo 5: Análise (SEMPRE VISÍVEL) ---
st.header("Passo 5: Análise dos Dados Processados", divider="rainbow")
st.markdown("Esta seção mostra a análise do arquivo CSV que acabamos de criar. Use-a para verificar se a distribuição das classes (o balanceamento) está como você esperava.")
display_report()