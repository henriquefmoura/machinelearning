# 🔬 Guia MLflow — Rastreamento de Experimentos ML

O app agora integra **MLflow** para rastrear automaticamente todos os experimentos de machine learning!

## ✨ O que é rastreado?

### Cada experimento registra:
- **Parâmetros**: tipo de problema, coluna alvo, número de features, clusters, etc.
- **Métricas**:
  - Classificação: acurácia (teste + cross-validação) para cada modelo
  - Regressão: MAE e R² para cada modelo
  - Clustering: inércia do KMeans
- **Melhor modelo**: nome e métrica do modelo que venceu

### Modelos testados:
- **Classificação**: Regressão Logística, Random Forest, Gradient Boosting, Rede Neural (PyTorch)
- **Regressão**: Regressão Linear, Random Forest, Gradient Boosting, Rede Neural (PyTorch)
- **Clustering**: KMeans (número de clusters configurável)

## 📊 Como visualizar os experimentos

### Opção 1: Interface web MLflow (Recomendado)

Execute no terminal:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

Depois acesse: **http://localhost:5000**

Lá você pode:
- ✅ Ver histórico de todos os experimentos rodados
- ✅ Comparar métricas entre modelos diferentes
- ✅ Visualizar parâmetros e configurações
- ✅ Exportar dados para análise

### Opção 2: Dentro do Streamlit

No final da seção de ML, após rodar uma análise, aparecerá um botão azul com instruções para abrir o MLflow.

## 📁 Onde estão os dados?

Os experimentos são salvos em:
```
./mlruns/
├── 0/                    # Experimento padrão
├── 1/                    # ML_Classificacao_20260505_...
├── 2/                    # ML_Regressao_20260505_...
└── ...
```

Cada pasta contém:
- `meta.yaml` — nome e timestamp do experimento
- `runs/` — dados de cada execução (métricas, parâmetros)

## 🎯 Exemplos de uso

### Cenário 1: Comparar dois uploads diferentes
1. Upload 1 → roda ML → MLflow rastreia "Experimento A"
2. Upload 2 → roda ML → MLflow rastreia "Experimento B"
3. Abra MLflow UI e compare as métricas lado-a-lado

### Cenário 2: Entender o melhor modelo
1. Rode a análise com seus dados
2. Note qual modelo teve a melhor métrica (mostrará na UI)
3. Abra MLflow para ver todos os parâmetros e configurações

### Cenário 3: Reproduzir resultados
1. Abra MLflow UI
2. Procure o run que quer reproduzir
3. Veja exatamente quais parâmetros foram usados
4. Use os mesmos dados e configurações

## ⚙️ Configuração técnica

MLflow está configurado para usar **backend local SQLite** (não requer servidor externo):

```python
mlflow.set_tracking_uri("file:./mlruns")  # Armazena em disk local
```

Isso significa:
- ✅ Funciona offline (sem internet)
- ✅ Dados persistem após fechamento do app
- ✅ Sem necessidade de conta em cloud
- ✅ Rápido e simples

## 🚀 Próximas melhorias

- [ ] Logar modelos serializados (.pkl, .pt) para reprodução
- [ ] Adicionar hyperparameter tuning com MLflow Optuna
- [ ] Criar pipeline de retrainamento automático
- [ ] Integrar com DVC (Data Version Control) para versionamento de dados

## 📖 Referências

- [Documentação MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/index.html)
- [MLflow UI Tutorial](https://mlflow.org/docs/latest/tracking/index.html#tracking-ui)
