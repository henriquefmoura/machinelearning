import traceback

import streamlit as st


# Entry point unico: executa o app principal em ml_insights.
# Se houver erro de import/runtime, mostra traceback na tela para facilitar diagnostico no Cloud.
try:
    import ml_insights  # noqa: F401
except Exception as exc:
    st.set_page_config(page_title="ML Insights Hub - erro", page_icon="⚠️", layout="centered")
    st.title("⚠️ Falha ao iniciar o app")
    st.error("Nao foi possivel carregar ml_insights.py")
    st.exception(exc)
    st.code(traceback.format_exc())
