import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Previsão de Contratação", page_icon="🤖", layout="centered")

st.title("🤖 Previsão de Contratação de Clientes")
st.markdown("Modelo de **Regressão Logística** para prever se um cliente vai contratar ou não.")

# --- Dados base ---
dados = pd.DataFrame({
    'preco':        [100, 200, 150, 300, 250, 180],
    'complexidade': [1,   2,   1,   3,   2,   1],
    'canal_digital':[0,   1,   0,   1,   1,   0],
})
dados['contratou'] = (
    (dados['complexidade'] > 1) &
    (dados['preco'] < 350) &
    (dados['canal_digital'] == 1)
).astype(int)

X = dados[['preco', 'complexidade']]
y = dados['contratou']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)
acuracia = accuracy_score(y_test, modelo.predict(X_test))

# --- Métricas do modelo ---
st.subheader("📊 Desempenho do Modelo")
col1, col2 = st.columns(2)
col1.metric("Acurácia", f"{acuracia:.0%}")
col2.metric("Algoritmo", "Regressão Logística")

# --- Importância das variáveis ---
st.subheader("📌 Importância das Variáveis")
coef = pd.DataFrame({
    'Variável': X.columns,
    'Peso (coeficiente)': modelo.coef_[0]
})
st.bar_chart(coef.set_index('Variável'))

# --- Ranking de clientes ---
st.subheader("🏆 Ranking de Clientes por Probabilidade")
dados['probabilidade'] = modelo.predict_proba(X)[:, 1]
ranking = dados.sort_values('probabilidade', ascending=False).reset_index(drop=True)
ranking.index += 1
st.dataframe(ranking.style.format({'probabilidade': '{:.1%}'}), use_container_width=True)

# --- Previsão para novo cliente ---
st.subheader("🔍 Prever Novo Cliente")
col_a, col_b = st.columns(2)
with col_a:
    preco = st.slider("Preço (R$)", min_value=50, max_value=500, value=220, step=10)
with col_b:
    complexidade = st.selectbox("Complexidade", [1, 2, 3], index=1)

if st.button("Calcular Probabilidade", type="primary"):
    novo = pd.DataFrame({'preco': [preco], 'complexidade': [complexidade]})
    prob = modelo.predict_proba(novo)[0]
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("✅ Probabilidade de Contratar", f"{prob[1]:.1%}")
    c2.metric("❌ Probabilidade de Não Contratar", f"{prob[0]:.1%}")

    if prob[1] >= 0.5:
        st.success("Este cliente tem **alta chance** de contratar!")
    else:
        st.warning("Este cliente tem **baixa chance** de contratar.")
