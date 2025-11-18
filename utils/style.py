import streamlit as st

def load_custom_css(css_file_path):
    with open(css_file_path) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)