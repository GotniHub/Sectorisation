import streamlit as st

# Configuration de la page - doit être le premier appel Streamlit
st.set_page_config(
    page_title="Sectorisation Commerciale",
    page_icon="📊",
    layout="wide"
)

#logo
st.logo("LOGO.png", icon_image="Logom.png")

st.write("# Welcome, C'est Advent+ Africa! 👋")
st.image("LOGO.png", use_column_width=True)