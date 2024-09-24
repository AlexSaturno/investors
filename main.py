################################################################################################################################
# Bibliotecas
################################################################################################################################
from posthog import page
import streamlit as st

from utils import *

################################################################################################################################
# UX
################################################################################################################################
# Inicio da aplicação
st.set_page_config(
    page_title="Investors",
    page_icon=":black_medium_square:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Leitura do arquivo css de estilização
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# List of pages
pages = {
    "01_Investors": "./pages/01_Investors.py",
    "02_Traders": "./pages/02_Traders.py",
}


################################################################################################################################
# UI
################################################################################################################################
# Inicio da aplicação
def main():
    st.subheader("Página inicial", divider=True)
    investors_button = st.button("Investors", key="button_investors")
    traders_button = st.button("Traders", key="button_traders")

    if investors_button:
        st.switch_page(pages["01_Investors"])
    if traders_button:
        st.switch_page(pages["02_Traders"])


if __name__ == "__main__":
    main()
