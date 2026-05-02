import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import requests
import json
import hmac

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error, r2_score
)
import re
import os
import io
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="ML Insights Hub", page_icon="📊", layout="wide")

HUB_FILE = Path("hub_dados.parquet")
HUB_KEY_FILE = Path("hub_key.txt")
DEFAULT_APP_PASSWORD = "mlhub123"
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", DEFAULT_APP_PASSWORD))

# Detecta se está rodando na nuvem (Streamlit Cloud) ou local
IS_CLOUD = not HUB_FILE.parent.exists() or os.environ.get("STREAMLIT_SERVER_HEADLESS") == "1"


def salvar_hub(df, key):
    """Salva em disco quando local; na nuvem usa apenas session_state."""
    try:
        df.to_parquet(HUB_FILE, index=False)
        HUB_KEY_FILE.write_text(key)
    except Exception:
        pass  # nuvem sem disco persistente — OK, dados ficam na sessão


def carregar_hub():
    df = pd.DataFrame()
    key = "id_cliente"
    if HUB_FILE.exists():
        try:
            df = pd.read_parquet(HUB_FILE)
        except Exception:
            df = pd.DataFrame()
    if HUB_KEY_FILE.exists():
        key = HUB_KEY_FILE.read_text().strip()
    return df, key


def hub_para_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.read()


def build_demo_hub():
    return pd.DataFrame(
        {
            "id_cliente": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "nome": [
                "Cliente A",
                "Cliente B",
                "Cliente C",
                "Cliente D",
                "Cliente E",
                "Cliente F",
                "Cliente G",
                "Cliente H",
                "Cliente I",
                "Cliente J",
            ],
            "preco": [120, 240, 180, 320, 280, 150, 350, 210, 260, 190],
            "complexidade": [1, 2, 1, 3, 2, 1, 3, 2, 2, 1],
            "canal_digital": [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            "contratou": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            "cidade": [
                "Sao Paulo",
                "Campinas",
                "Sao Paulo",
                "Rio de Janeiro",
                "Curitiba",
                "Santos",
                "Belo Horizonte",
                "Campinas",
                "Recife",
                "Porto Alegre",
            ],
        }
    )


def require_authentication():
    if st.session_state.get("auth_ok", False):
        return

    st.title("Acesso protegido")
    st.write("Digite a senha para entrar no painel.")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar", type="primary"):
        if hmac.compare_digest(senha, APP_PASSWORD):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Senha incorreta.")

    st.info("Dica: no Streamlit Cloud, defina APP_PASSWORD em Settings > Secrets.")
    st.stop()


require_authentication()

st.title("ML Insights Hub")
st.markdown(
    "Recebe planilhas em formatos diferentes, organiza automaticamente e treina modelos de ML."
)

if st.sidebar.button("Sair"):
    st.session_state.auth_ok = False
    st.rerun()


SYNONYMS = {
    "id_cliente": ["id_cliente", "id", "cliente_id", "codigo_cliente", "cod_cliente", "customer_id"],
    "nome": ["nome", "cliente", "nome_cliente", "razao_social"],
    "email": ["email", "e_mail", "mail"],
    "telefone": ["telefone", "fone", "celular", "whatsapp"],
    "idade": ["idade", "anos"],
    "sexo": ["sexo", "genero"],
    "cidade": ["cidade", "municipio"],
    "estado": ["estado", "uf"],
    "canal_digital": ["canal_digital", "canal", "digital", "canal_online", "origem_digital"],
    "preco": ["preco", "valor", "ticket", "preco_medio", "valor_medio", "faturamento"],
    "complexidade": ["complexidade", "nivel_complexidade", "score_complexidade"],
    "contratou": ["contratou", "converteu", "compra", "comprou", "fechou", "target", "y"],
}


def normalize_name(name):
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def canonical_column(col):
    norm = normalize_name(col)
    for target, aliases in SYNONYMS.items():
        if norm == target or norm in aliases:
            return target
    return norm


def coerce_numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = (
        series.astype(str)
        .str.replace("R$", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    out = pd.to_numeric(cleaned, errors="coerce")
    if out.notna().sum() >= max(5, int(0.6 * len(series))):
        return out
    return series


def preprocess_df(raw_df):
    df = raw_df.copy()
    original_cols = list(df.columns)
    mapped_cols = [canonical_column(c) for c in df.columns]

    used = {}
    final_cols = []
    for c in mapped_cols:
        n = used.get(c, 0)
        if n == 0:
            final_cols.append(c)
        else:
            final_cols.append(f"{c}_{n+1}")
        used[c] = n + 1

    df.columns = final_cols
    for c in df.columns:
        df[c] = coerce_numeric_series(df[c])

    return df, original_cols, final_cols


def upsert_hub(base_df, new_df, key_col):
    if base_df is None or base_df.empty:
        return new_df.copy()

    all_cols = sorted(set(base_df.columns).union(set(new_df.columns)))
    left = base_df.reindex(columns=all_cols)
    right = new_df.reindex(columns=all_cols)

    left_idx = left.set_index(key_col)
    right_idx = right.set_index(key_col)

    merged = right_idx.combine_first(left_idx)
    return merged.reset_index()


def suggest_target(df):
    for c in ["contratou", "target", "y"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def make_binary(y):
    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.Series(y).dropna().unique().tolist())
        if len(uniq) <= 2:
            return y.astype(int), None
    le = LabelEncoder()
    return le.fit_transform(y.astype(str)), le


def build_data_summary(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    summary = {
        "registros": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "faltantes_total": int(df.isna().sum().sum()),
        "colunas_numericas": numeric_cols,
        "colunas_categoricas": cat_cols,
        "top_missing": (
            df.isna().sum().sort_values(ascending=False).head(8).to_dict()
        ),
    }

    if numeric_cols:
        stats = df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(3)
        summary["estatisticas_numericas"] = stats.to_dict(orient="index")

    return summary


def local_insights(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    lines = []

    lines.append(f"- Base consolidada com {df.shape[0]} registros e {df.shape[1]} colunas.")
    missing_pct = (df.isna().sum().sum() / (df.shape[0] * max(1, df.shape[1]))) * 100
    lines.append(f"- Taxa geral de valores ausentes: {missing_pct:.1f}%.")

    if numeric_cols:
        variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
        top_var = variances.head(3).index.tolist()
        lines.append(f"- Colunas mais variáveis: {', '.join(top_var)}.")

        corr = df[numeric_cols].corr(numeric_only=True)
        if corr.shape[0] > 1:
            corr_pairs = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1:]:
                    corr_pairs.append((c1, c2, abs(corr.loc[c1, c2]), corr.loc[c1, c2]))
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            best = corr_pairs[0]
            lines.append(
                f"- Correlação mais forte: {best[0]} x {best[1]} = {best[3]:.2f}."
            )

    lines.append("- Recomendação: use a coluna mais crítica de negócio como alvo e rode classificação/regressão.")
    return "\n".join(lines)


def ai_insights_openai(api_key, model_name, user_prompt, summary):
    url = "https://api.openai.com/v1/responses"
    system_prompt = (
        "Voce e um analista sênior de dados e machine learning. "
        "Entregue insights objetivos em portugues brasileiro, com foco em negocio e acoes praticas. "
        "Responda em markdown com secoes curtas: Diagnostico, Oportunidades, Riscos, Proximos passos."
    )

    payload = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Pergunta do usuario: {user_prompt}\n\n"
                            f"Resumo dos dados (json):\n{json.dumps(summary, ensure_ascii=True)}"
                        ),
                    }
                ],
            },
        ],
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    if "output_text" in data and data["output_text"]:
        return data["output_text"]

    try:
        parts = []
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
        return "\n".join([p for p in parts if p]).strip()
    except Exception:
        return "Nao foi possivel extrair resposta textual da API."

# 1. Hub de ingestao incremental
st.sidebar.markdown("### Hub de Ingestao")
uploaded_files = st.sidebar.file_uploader(
    "Adicionar planilhas (Excel/CSV)",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True,
)

if "rpa_panel_open" not in st.session_state:
    st.session_state.rpa_panel_open = False
if "rpa_last_run" not in st.session_state:
    st.session_state.rpa_last_run = None

if st.sidebar.button("Versao RPA"):
    st.session_state.rpa_panel_open = not st.session_state.rpa_panel_open

if "hub_df" not in st.session_state:
    _df_disk, _key_disk = carregar_hub()
    st.session_state.hub_df = _df_disk
    st.session_state.hub_key = _key_disk
    if not _df_disk.empty:
        st.sidebar.info(f"Hub restaurado do disco: {len(_df_disk)} clientes")

# ── Restaurar hub de outro ambiente (upload de .parquet) ──────
st.sidebar.markdown("---")
st.sidebar.markdown("**Restaurar hub salvo**")
hub_restore = st.sidebar.file_uploader(
    "Importar hub_dados.parquet",
    type=["parquet"],
    key="hub_restore",
)
if hub_restore is not None:
    try:
        _restored = pd.read_parquet(hub_restore)
        st.session_state.hub_df = _restored
        salvar_hub(_restored, st.session_state.get("hub_key", "id_cliente"))
        st.sidebar.success(f"{len(_restored)} clientes restaurados!")
    except Exception as e:
        st.sidebar.error(f"Erro ao restaurar: {e}")
st.sidebar.markdown("---")

if uploaded_files:
    first = uploaded_files[0]
    if first.name.lower().endswith(".csv"):
        temp_df = pd.read_csv(first)
    else:
        temp_df = pd.read_excel(first)
    temp_df, _, mapped = preprocess_df(temp_df)

    candidates = [c for c in temp_df.columns if "id" in c or c.endswith("_id")]
    default_key = "id_cliente" if "id_cliente" in temp_df.columns else (candidates[0] if candidates else temp_df.columns[0])

    key_col = st.sidebar.selectbox(
        "Chave unica do cliente (para consolidar updates)",
        options=temp_df.columns.tolist(),
        index=temp_df.columns.tolist().index(default_key),
    )

    ingest_now = st.sidebar.button("Ingerir no Hub", type="primary")
    if ingest_now:
        hub = st.session_state.hub_df.copy()
        rows_before = len(hub)
        run_details = []
        for f in uploaded_files:
            if f.name.lower().endswith(".csv"):
                raw = pd.read_csv(f)
            else:
                raw = pd.read_excel(f)
            clean, original_cols, final_cols = preprocess_df(raw)
            if key_col not in clean.columns:
                st.warning(f"Arquivo {f.name} ignorado: chave {key_col} nao encontrada.")
                run_details.append(
                    {
                        "arquivo": f.name,
                        "status": "ignorado",
                        "linhas_lidas": int(len(raw)),
                        "linhas_validas": 0,
                        "colunas": int(raw.shape[1]),
                    }
                )
                continue

            clean = clean.dropna(subset=[key_col]).drop_duplicates(subset=[key_col], keep="last")
            linhas_validas = int(len(clean))
            hub = upsert_hub(hub, clean, key_col)
            run_details.append(
                {
                    "arquivo": f.name,
                    "status": "processado",
                    "linhas_lidas": int(len(raw)),
                    "linhas_validas": linhas_validas,
                    "colunas": int(clean.shape[1]),
                }
            )

            with st.expander(f"Mapeamento automatico: {f.name}"):
                st.write(pd.DataFrame({"original": original_cols, "padrao": final_cols}))

        st.session_state.hub_df = hub
        st.session_state.hub_key = key_col
        salvar_hub(hub, key_col)
        processed_count = sum(1 for d in run_details if d["status"] == "processado")
        ignored_count = sum(1 for d in run_details if d["status"] == "ignorado")
        rows_after = len(hub)
        st.session_state.rpa_last_run = {
            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "versao": f"RPA-{datetime.now().strftime('%Y.%m.%d.%H%M')}",
            "chave": key_col,
            "arquivos_recebidos": len(uploaded_files),
            "arquivos_processados": processed_count,
            "arquivos_ignorados": ignored_count,
            "linhas_antes": int(rows_before),
            "linhas_depois": int(rows_after),
            "delta_linhas": int(rows_after - rows_before),
            "detalhes": run_details,
        }
        ts = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.success(f"Dados consolidados e salvos em disco — {ts}")
        st.rerun()

if st.sidebar.button("Limpar Hub"):
    st.session_state.hub_df = pd.DataFrame()
    if HUB_FILE.exists():
        HUB_FILE.unlink()
    if HUB_KEY_FILE.exists():
        HUB_KEY_FILE.unlink()

df = st.session_state.hub_df.copy()
is_demo_mode = False

if df.empty:
    df = build_demo_hub()
    is_demo_mode = True

st.sidebar.success(f"Hub ativo: {len(df)} clientes")
st.sidebar.markdown(f"Colunas consolidadas: {df.shape[1]}")

if is_demo_mode:
    st.info(
        "Modo demonstracao ativo: exibindo dashboard com dados exemplo. "
        "Para usar seus dados reais, envie planilhas na barra lateral e clique em Ingerir no Hub."
    )

# ─────────────────────────────────────────────
# 3. VISÃO GERAL DOS DADOS
# ─────────────────────────────────────────────
with st.expander("Visao geral do hub", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", df.shape[0])
    col2.metric("Colunas", df.shape[1])
    col3.metric("Valores ausentes", int(df.isnull().sum().sum()))

    st.dataframe(df.head(10), width="stretch")

    st.markdown("Estatisticas descritivas")
    desc = df.describe(include="all").T
    st.dataframe(desc, width="stretch")

# ─────────────────────────────────────────────
# DASHBOARD FIXO — experiência executiva
# ─────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols and df[c].nunique() <= 30]

st.markdown("## Dashboard Executivo")
st.caption("Painel fixo com visão de negócio e exploração rápida dos dados consolidados.")

if st.session_state.rpa_panel_open:
    st.markdown("### Versao RPA")
    rpa = st.session_state.rpa_last_run
    if not rpa:
        st.info(
            "Nenhuma execução RPA registrada ainda. Envie planilhas e clique em Ingerir no Hub para preencher esta area."
        )
    else:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Versao", rpa["versao"])
        r2.metric("Arquivos processados", f"{rpa['arquivos_processados']}/{rpa['arquivos_recebidos']}")
        r3.metric("Delta de linhas", f"{rpa['delta_linhas']:+,}")
        r4.metric("Chave de consolidacao", rpa["chave"])

        st.caption(f"Ultima execução: {rpa['timestamp']}")
        st.markdown("**Itens trazidos pelo RPA**")
        st.markdown(
            "\n".join(
                [
                    f"- Arquivos ignorados: {rpa['arquivos_ignorados']}",
                    f"- Registros antes/depois: {rpa['linhas_antes']} -> {rpa['linhas_depois']}",
                    "- Status da execução: concluida",
                ]
            )
        )

        st.dataframe(pd.DataFrame(rpa["detalhes"]), width="stretch")

missing_total = int(df.isnull().sum().sum())
missing_pct = (missing_total / (max(1, df.shape[0] * df.shape[1]))) * 100

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Clientes", f"{df.shape[0]:,}")
kpi_2.metric("Atributos", f"{df.shape[1]:,}")
kpi_3.metric("Valores ausentes", f"{missing_total:,}")
kpi_4.metric("Qualidade preenchimento", f"{100 - missing_pct:.1f}%")

tab_overview, tab_profile, tab_rel, tab_export = st.tabs([
    "Visão Geral",
    "Perfil dos Dados",
    "Relações e Tendências",
    "Exportação",
])

with tab_overview:
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("### Amostra do Hub")
        st.dataframe(df.head(15), width="stretch")
    with right:
        st.markdown("### Estatísticas rápidas")
        if numeric_cols:
            summary = df[numeric_cols].describe().T[["mean", "std", "min", "max"]].round(2)
            st.dataframe(summary, width="stretch")
        else:
            st.info("Nenhuma coluna numérica encontrada para estatísticas.")

    if numeric_cols:
        st.markdown("### Indicadores por variável")
        kpi_cols = numeric_cols[:6]
        kpi_grid = st.columns(len(kpi_cols))
        for i, c in enumerate(kpi_cols):
            total = df[c].sum()
            media = df[c].mean()
            kpi_grid[i].metric(
                label=c.replace("_", " ").title(),
                value=f"{media:,.2f}",
                delta=f"total {total:,.0f}",
            )

with tab_profile:
    if numeric_cols:
        st.markdown("### Distribuição das variáveis numéricas")
        n = len(numeric_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig_dist, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, c in enumerate(numeric_cols):
            axes[i].hist(df[c].dropna(), bins=20, color="#4e8cff", edgecolor="white", alpha=0.85)
            axes[i].set_title(c.replace("_", " ").title(), fontsize=10)
            axes[i].set_ylabel("Frequência")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)
        plt.close()

    if cat_cols:
        st.markdown("### Distribuição das categorias")
        ccol1, ccol2 = st.columns(2)
        for idx, c in enumerate(cat_cols[:6]):
            vcounts = df[c].value_counts().head(12)
            fig_cat, ax_cat = plt.subplots(figsize=(7, 3.2))
            ax_cat.barh(vcounts.index.astype(str), vcounts.values, color="#2ecc71", edgecolor="white")
            ax_cat.set_title(c.replace("_", " ").title(), fontsize=10)
            ax_cat.set_xlabel("Qtd.")
            ax_cat.invert_yaxis()
            fig_cat.tight_layout()
            if idx % 2 == 0:
                ccol1.pyplot(fig_cat)
            else:
                ccol2.pyplot(fig_cat)
            plt.close()

with tab_rel:
    if len(numeric_cols) >= 2:
        st.markdown("### Mapa de correlação")
        corr = df[numeric_cols].corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) - 1)))
        cax = ax_corr.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig_corr.colorbar(cax)
        ax_corr.set_xticks(range(len(corr.columns)))
        ax_corr.set_yticks(range(len(corr.columns)))
        ax_corr.set_xticklabels([c.replace("_", " ") for c in corr.columns], rotation=45, ha="left", fontsize=9)
        ax_corr.set_yticklabels([c.replace("_", " ") for c in corr.columns], fontsize=9)
        for (i, j), val in np.ndenumerate(corr.values):
            ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(val) > 0.5 else "black")
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        plt.close()

        st.markdown("### Relação entre variáveis")
        rel1, rel2, rel3 = st.columns(3)
        col_x = rel1.selectbox("Eixo X", numeric_cols, index=0, key="dash_x")
        col_y = rel2.selectbox("Eixo Y", numeric_cols, index=min(1, len(numeric_cols) - 1), key="dash_y")
        color_by = rel3.selectbox("Colorir por", ["Nenhum"] + cat_cols, key="dash_color")

        fig_sc, ax_sc = plt.subplots(figsize=(8.6, 4.8))
        if color_by != "Nenhum":
            for grupo, sub in df.groupby(color_by):
                ax_sc.scatter(sub[col_x], sub[col_y], label=str(grupo), alpha=0.7, s=40)
            ax_sc.legend(title=color_by.replace("_", " ").title(), fontsize=8)
        else:
            ax_sc.scatter(df[col_x], df[col_y], alpha=0.6, color="#9b59b6", s=40)
        ax_sc.set_xlabel(col_x.replace("_", " ").title())
        ax_sc.set_ylabel(col_y.replace("_", " ").title())
        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        plt.close()

    date_cols = [c for c in df.columns if "data" in c.lower() or "date" in c.lower() or "mes" in c.lower()]
    if date_cols and numeric_cols:
        st.markdown("### Evolução temporal")
        dcol = date_cols[0]
        vcol = st.selectbox("Variável temporal", numeric_cols, key="ts_var")
        try:
            df_ts = df[[dcol, vcol]].dropna().copy()
            df_ts[dcol] = pd.to_datetime(df_ts[dcol], dayfirst=True, errors="coerce")
            df_ts = df_ts.dropna(subset=[dcol]).sort_values(dcol)
            df_ts_g = df_ts.groupby(dcol)[vcol].mean()
            fig_ts, ax_ts = plt.subplots(figsize=(9, 4))
            ax_ts.plot(df_ts_g.index, df_ts_g.values, color="#e74c3c", linewidth=2)
            ax_ts.fill_between(df_ts_g.index, df_ts_g.values, alpha=0.15, color="#e74c3c")
            ax_ts.set_xlabel("Data")
            ax_ts.set_ylabel(vcol.replace("_", " ").title())
            fig_ts.tight_layout()
            st.pyplot(fig_ts)
            plt.close()
        except Exception:
            st.info("Coluna de data encontrada, mas não foi possível interpretar os valores.")

with tab_export:
    st.markdown("### Exportar base consolidada")
    st.write("Baixe a base tratada para usar em outros ambientes ou restaurar no hub.")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hub")
    buf.seek(0)

    exp1, exp2 = st.columns(2)
    exp1.download_button(
        "Baixar Excel consolidado",
        data=buf,
        file_name="hub_clientes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    exp2.download_button(
        "Baixar Parquet consolidado",
        data=hub_para_bytes(df),
        file_name="hub_dados.parquet",
        mime="application/octet-stream",
    )

with st.expander("Assistente IA para Insights", expanded=False):
    st.markdown("Use IA para gerar leitura executiva dos dados consolidados.")

    c1, c2 = st.columns([2, 1])
    with c1:
        ai_prompt = st.text_area(
            "Pergunta para a IA",
            value="Quais sao os principais insights, riscos e oportunidades comerciais dessa base?",
            height=90,
        )
    with c2:
        ai_model = st.text_input("Modelo", value="gpt-4o-mini")
        use_api = st.checkbox("Usar API externa", value=False)

    api_key = ""
    if use_api:
        api_key = st.text_input("OPENAI_API_KEY", type="password")

    if st.button("Gerar insights com IA"):
        summary = build_data_summary(df)

        if use_api and api_key:
            try:
                with st.spinner("Consultando IA..."):
                    answer = ai_insights_openai(api_key, ai_model, ai_prompt, summary)
                st.markdown(answer)
            except Exception as exc:
                st.error(f"Falha ao consultar API: {exc}")
                st.info("Exibindo insights locais como fallback.")
                st.markdown(local_insights(df))
        else:
            st.info("Modo local ativo (sem API).")
            st.markdown(local_insights(df))

# ─────────────────────────────────────────────
# 4. PREPARAÇÃO DAS COLUNAS
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Configuracao do modelo")

colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
colunas_todas = df.columns.tolist()

target_col = st.sidebar.selectbox(
    "Coluna alvo",
    options=colunas_todas,
    index=colunas_todas.index(suggest_target(df))
)

features_disponiveis = [c for c in colunas_numericas if c != target_col]
features_selecionadas = st.sidebar.multiselect(
    "Variaveis para prever",
    options=features_disponiveis,
    default=features_disponiveis
)

tipo_problema = st.sidebar.radio(
    "Tipo de analise",
    ["Classificação (sim/não)", "Regressão (valor numérico)", "Clustering (agrupamento)"]
)

rodar = st.sidebar.button("Rodar analise")

if not rodar:
    st.info("Configure na barra lateral e clique em Rodar analise.")
    st.stop()

if len(features_selecionadas) == 0:
    st.error("Selecione ao menos uma variavel numerica preditora.")
    st.stop()

# ─────────────────────────────────────────────
# 5. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
df_model = df[features_selecionadas + [target_col]].dropna()
X = df_model[features_selecionadas]

le = LabelEncoder()
scaler = StandardScaler()

# ─────────────────────────────────────────────
# 6. ANÁLISE — CLASSIFICAÇÃO
# ─────────────────────────────────────────────
if tipo_problema == "Classificação (sim/não)":
    st.header("Classificacao de clientes")

    y_raw = df_model[target_col]
    y, _ = make_binary(y_raw)

    if len(np.unique(y)) < 2:
        st.error("A coluna alvo precisa ter ao menos 2 classes para classificacao.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    resultados = []
    for nome, m in modelos.items():
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        cv_folds = min(5, max(2, len(y) // 5))
        cv = cross_val_score(m, X, y, cv=cv_folds, scoring="accuracy").mean()
        resultados.append({"Modelo": nome, "Acurácia (teste)": acc, "Acurácia (cross-val)": cv})

    df_res = pd.DataFrame(resultados).sort_values("Acurácia (teste)", ascending=False)
    melhor_nome = df_res.iloc[0]["Modelo"]
    melhor_modelo = modelos[melhor_nome]

    st.subheader("Comparacao de modelos")
    st.dataframe(df_res.style.format({"Acurácia (teste)": "{:.1%}", "Acurácia (cross-val)": "{:.1%}"}), width="stretch")
    st.success(f"✅ Melhor modelo: **{melhor_nome}** com **{df_res.iloc[0]['Acurácia (teste)']:.1%}** de acurácia")

    # Importância das variáveis
    if hasattr(melhor_modelo, "feature_importances_"):
        imp = pd.DataFrame({
            "Variável": features_selecionadas,
            "Importância": melhor_modelo.feature_importances_
        }).sort_values("Importância", ascending=False)

        st.subheader("Variaveis mais importantes")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp["Variável"], imp["Importância"], color="#4e8cff")
        ax.set_xlabel("Importância")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

        st.markdown("Insight: as variaveis acima tem maior impacto na previsao.")

    # Ranking de probabilidades
    probs = melhor_modelo.predict_proba(X)[:, 1]
    df_rank = df_model.copy()
    df_rank["prob_contratar"] = probs
    df_rank["previsão"] = ["✅ Contratar" if p >= 0.5 else "❌ Não contratar" for p in probs]
    df_rank = df_rank.sort_values("prob_contratar", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    st.subheader("Ranking por probabilidade de contratar")
    st.dataframe(
        df_rank.style.format({"prob_contratar": "{:.1%}"}),
        width="stretch"
    )

    # Previsão interativa
    st.subheader("Simular novo cliente")
    inputs = {}
    cols = st.columns(len(features_selecionadas))
    for i, feat in enumerate(features_selecionadas):
        min_v = float(df[feat].min())
        max_v = float(df[feat].max())
        med_v = float(df[feat].median())
        inputs[feat] = cols[i].slider(feat, min_value=min_v, max_value=max_v, value=med_v)

    novo = pd.DataFrame([inputs])
    prob_novo = melhor_modelo.predict_proba(novo)[0]
    c1, c2 = st.columns(2)
    c1.metric("✅ Probabilidade de Contratar", f"{prob_novo[1]:.1%}")
    c2.metric("❌ Probabilidade de Não Contratar", f"{prob_novo[0]:.1%}")
    if prob_novo[1] >= 0.5:
        st.success("Perfil com alta chance de contratar.")
    else:
        st.warning("Perfil com baixa chance de contratar.")

# ─────────────────────────────────────────────
# 7. ANÁLISE — REGRESSÃO
# ─────────────────────────────────────────────
elif tipo_problema == "Regressão (valor numérico)":
    st.header("Regressao - previsao de valor")

    y = df_model[target_col].values

    if len(df_model) < 10:
        st.warning("Poucos dados para regressao. Ideal: pelo menos 30 registros.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    modelos = {
        "Regressão Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    resultados = []
    for nome, m in modelos.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        resultados.append({"Modelo": nome, "MAE (erro médio)": mae, "R² (ajuste)": r2})

    df_res = pd.DataFrame(resultados).sort_values("R² (ajuste)", ascending=False)
    melhor_nome = df_res.iloc[0]["Modelo"]
    melhor_modelo = modelos[melhor_nome]

    st.subheader("Comparacao de modelos")
    st.dataframe(df_res.style.format({"MAE (erro médio)": "{:.2f}", "R² (ajuste)": "{:.2f}"}), width="stretch")
    st.success(f"✅ Melhor modelo: **{melhor_nome}** com R² = **{df_res.iloc[0]['R² (ajuste)']:.2f}**")

    if hasattr(melhor_modelo, "feature_importances_"):
        imp = pd.DataFrame({
            "Variável": features_selecionadas,
            "Importância": melhor_modelo.feature_importances_
        }).sort_values("Importância", ascending=False)

        st.subheader("Variaveis mais importantes")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp["Variável"], imp["Importância"], color="#f4a261")
        ax.set_xlabel("Importância")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    # Previsão vs Real
    pred_test = melhor_modelo.predict(X_test)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(y_test, pred_test, alpha=0.6, color="#4e8cff")
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax2.set_xlabel("Valor Real")
    ax2.set_ylabel("Valor Previsto")
    ax2.set_title("Real vs Previsto")
    st.subheader("Real vs previsto")
    st.pyplot(fig2)
    plt.close()

    # Ranking
    previsoes_todas = melhor_modelo.predict(X)
    df_rank = df_model.copy()
    df_rank[f"previsão_{target_col}"] = previsoes_todas
    df_rank = df_rank.sort_values(f"previsão_{target_col}", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    st.subheader(f"Ranking por {target_col} previsto")
    st.dataframe(df_rank, width="stretch")

# ─────────────────────────────────────────────
# 8. ANÁLISE — CLUSTERING
# ─────────────────────────────────────────────
elif tipo_problema == "Clustering (agrupamento)":
    st.header("Clustering - grupos de clientes")

    X_scaled = scaler.fit_transform(X)

    n_clusters = st.slider("Numero de grupos", min_value=2, max_value=8, value=3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_cluster = df_model.copy()
    df_cluster["Grupo"] = [f"Grupo {l+1}" for l in labels]

    st.subheader("Perfil medio por grupo")
    perfil = df_cluster.groupby("Grupo")[features_selecionadas].mean().round(2)
    st.dataframe(perfil, width="stretch")

    st.subheader("Tamanho dos grupos")
    tamanhos = df_cluster["Grupo"].value_counts().reset_index()
    tamanhos.columns = ["Grupo", "Qtd. Clientes"]
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(tamanhos["Grupo"], tamanhos["Qtd. Clientes"], color=["#4e8cff", "#f4a261", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c", "#f39c12", "#e67e22"])
    ax3.set_ylabel("Quantidade de Clientes")
    st.pyplot(fig3)
    plt.close()

    if len(features_selecionadas) >= 2:
        st.subheader("Visualizacao dos grupos (2 variaveis)")
        fx = st.selectbox("Eixo X", features_selecionadas, index=0)
        fy = st.selectbox("Eixo Y", features_selecionadas, index=min(1, len(features_selecionadas)-1))
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        cores = ["#4e8cff", "#f4a261", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c", "#f39c12", "#e67e22"]
        for i, grupo in enumerate(sorted(df_cluster["Grupo"].unique())):
            sub = df_cluster[df_cluster["Grupo"] == grupo]
            ax4.scatter(sub[fx], sub[fy], label=grupo, alpha=0.7, color=cores[i % len(cores)])
        ax4.set_xlabel(fx)
        ax4.set_ylabel(fy)
        ax4.legend()
        st.pyplot(fig4)
        plt.close()

    st.subheader("Dados com grupos")
    st.dataframe(df_cluster, width="stretch")

    st.markdown("---")
    st.markdown("Insight: use os perfis por grupo para definir estrategias comerciais diferentes.")
