import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# base simples simulando clientes
dados = pd.DataFrame({
    'preco': [100, 200, 150, 300, 250, 180],
    'complexidade': [1, 2, 1, 3, 2, 1],
    'canal_digital': [0, 1, 0, 1, 1, 0],
    'contratou': [0, 1, 0, 1, 1, 0]
})

X = dados[['preco', 'complexidade']]
y = dados['contratou']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

print("Acurácia:", accuracy_score(y_test, previsoes))

novo_cliente = pd.DataFrame({
    'preco': [220],
    'complexidade': [2]
})

probabilidade = modelo.predict_proba(novo_cliente)

print("Probabilidade de não contratar:", probabilidade[0][0])
print("Probabilidade de contratar:", probabilidade[0][1])
dados['contratou'] = (
    (dados['complexidade'] > 1) & 
    (dados['preco'] < 350) & 
    (dados['canal_digital'] == 1)
).astype(int)
# importância das variáveis
import pandas as pd

coeficientes = pd.DataFrame({
    'variavel': X.columns,
    'peso': modelo.coef_[0]
})

print(coeficientes)
# probabilidade para todos os clientes
dados['probabilidade'] = modelo.predict_proba(X)[:,1]

# ordenar do maior para o menor
ranking = dados.sort_values(by='probabilidade', ascending=False)

print(ranking.head(10))

& "C:\Users\Henrique F. Moura\.local\bin\python3.14.exe" calculadora.py
