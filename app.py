import os as _os

# Executa ml_insights.py no mesmo contexto para compatibilidade total com Streamlit
_ml = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ml_insights.py")
exec(compile(open(_ml, encoding="utf-8").read(), _ml, "exec"))
