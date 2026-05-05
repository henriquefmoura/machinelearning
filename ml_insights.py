import sys
import traceback as _traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import hmac
import csv
import hashlib
from datetime import datetime

# Log de diagnóstico — aparece em Streamlit Cloud > Manage app > Logs
print(f"[ML_INSIGHTS] Python {sys.version} | numpy {np.__version__} | pandas {pd.__version__}", flush=True)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
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
import time
from pathlib import Path
try:
    import duckdb as _duckdb
    _DUCKDB_OK = True
except ImportError:
    _DUCKDB_OK = False

# MLflow desativado: não está em requirements.txt e causa segfault via conflito protobuf
# no Streamlit Cloud quando versões antigas ficam em cache.
# Para reativar localmente: instale mlflow manualmente e mude _MLFLOW_OK para True.
mlflow = None
_MLFLOW_OK = False

_TORCH_OK = False  # Lazy import — carrega sob demanda no bloco de ML
try:
    import pydeck as pdk
    _PYDECK_OK = True
except ImportError:
    _PYDECK_OK = False

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"], .stButton button, input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* KPI Cards */
    [data-testid="stMetric"] {
        background: #1e2530 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        transition: border-color 0.15s ease !important;
    }
    [data-testid="stMetric"]:hover { border-color: rgba(78,140,255,0.5) !important; }
    [data-testid="stMetricLabel"] p {
        font-size: 12px !important; font-weight: 600 !important;
        color: #a6a9b6 !important; text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px !important; font-weight: 700 !important; color: #fafafa !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #4e8cff !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
        transition: background 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3a73e6 !important;
        box-shadow: 0 4px 12px rgba(78,140,255,0.35) !important;
    }
    /* Secondary/default button */
    .stButton > button:not([kind="primary"]) {
        background: transparent !important;
        border: 1px solid #2d3748 !important; border-radius: 8px !important;
        color: #fafafa !important; font-weight: 500 !important;
        transition: border-color 0.15s ease, background 0.15s ease !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #4e8cff !important;
        background: rgba(78,140,255,0.06) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #2d3748 !important; gap: 0 !important; }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important; border: none !important;
        color: #a6a9b6 !important; font-weight: 500 !important;
        padding: 12px 20px !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4e8cff !important; border-bottom: 2px solid #4e8cff !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #fafafa !important; background: transparent !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #161b25 !important; border-right: 1px solid #2d3748 !important; }
    [data-testid="stSidebar"] h3 {
        font-size: 11px !important; font-weight: 700 !important;
        color: #6b7280 !important; text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px !important; overflow: hidden !important; border: 1px solid #2d3748 !important; }

    /* Alerts */
    [data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 4px !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1e2530 !important; border: 2px dashed #2d3748 !important;
        border-radius: 12px !important; transition: border-color 0.15s ease !important;
    }
    [data-testid="stFileUploader"]:hover { border-color: #4e8cff !important; }

    /* Dividers */
    hr { border-color: #2d3748 !important; }

    /* Hide Streamlit footer */
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)


import gc

def format_number(num, decimals=1, abbreviate=True):
    """Formata número inteligentemente: 1000 → 1.0K, 1000000 → 1.0M, etc"""
    try:
        num = float(num)
        if np.isnan(num) or np.isinf(num):
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"
    
    if not abbreviate or abs(num) < 1000:
        if isinstance(num, float):
            return f"{num:,.{decimals}f}".rstrip('0').rstrip('.')
        return f"{int(num):,}"
    
    for unit, divisor in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(num) >= divisor:
            return f"{(num / divisor):.{decimals}f}".rstrip('0').rstrip('.') + unit
    return f"{num:,.{decimals}f}".rstrip('0').rstrip('.')

st.set_page_config(page_title="ML Insights Hub", page_icon="📊", layout="wide")
inject_custom_css()

HUB_FILE = Path("hub_dados.parquet")
HUB_KEY_FILE = Path("hub_key.txt")
DEFAULT_APP_PASSWORD = "mlhub123"

try:
    HUB_DB_PATH = st.secrets.get("HUB_DB_PATH", os.environ.get("HUB_DB_PATH", "dados.duckdb"))
except Exception:
    HUB_DB_PATH = os.environ.get("HUB_DB_PATH", "dados.duckdb")

try:
    _raw_tbl = st.secrets.get("HUB_DB_TABLE", os.environ.get("HUB_DB_TABLE", "hub_auto"))
except Exception:
    _raw_tbl = os.environ.get("HUB_DB_TABLE", "hub_auto")

HUB_DB_TABLE = re.sub(r"[^a-zA-Z0-9_]", "_", str(_raw_tbl or "hub_auto")).strip("_") or "hub_auto"
if HUB_DB_TABLE[0].isdigit():
    HUB_DB_TABLE = f"_{HUB_DB_TABLE}"

try:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", DEFAULT_APP_PASSWORD))
except Exception:
    APP_PASSWORD = os.environ.get("APP_PASSWORD", DEFAULT_APP_PASSWORD)

# Detecta se está rodando na nuvem (Streamlit Cloud) ou local
try:
    IS_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "1"
except Exception:
    IS_CLOUD = False

APP_SCHEMA_VERSION = "2026-05-04-v8"
LARGE_FILE_THRESHOLD_MB = 100
MAX_DASHBOARD_ROWS = 200_000


def _open_hub_db():
    if not _DUCKDB_OK:
        return None
    try:
        if str(HUB_DB_PATH).startswith("md:"):
            try:
                md_token = st.secrets.get("MOTHERDUCK_TOKEN", os.environ.get("MOTHERDUCK_TOKEN", ""))
            except Exception:
                md_token = os.environ.get("MOTHERDUCK_TOKEN", "")
            if not md_token:
                return None
            return _duckdb.connect(HUB_DB_PATH, config={"motherduck_token": md_token})
        return _duckdb.connect(HUB_DB_PATH)
    except Exception as _db_exc:
        print(f"[ML_INSIGHTS] Falha ao abrir banco do hub ({HUB_DB_PATH}): {_db_exc}", flush=True)
        return None


def salvar_hub(df, key):
    """Salva hub em DuckDB (quando disponível) e mantém fallback em parquet."""
    try:
        df.to_parquet(HUB_FILE, index=False)
        HUB_KEY_FILE.write_text(key)
    except Exception:
        pass

    con = _open_hub_db()
    if con is None:
        return

    try:
        con.execute("CREATE TABLE IF NOT EXISTS hub_meta (k VARCHAR, v VARCHAR)")
        con.execute("DELETE FROM hub_meta WHERE k='hub_key'")
        con.execute("INSERT INTO hub_meta VALUES ('hub_key', ?)", [key])

        if df.empty:
            con.execute(f"DROP TABLE IF EXISTS {HUB_DB_TABLE}")
        else:
            con.register("_hub_tmp", df)
            con.execute(f"CREATE OR REPLACE TABLE {HUB_DB_TABLE} AS SELECT * FROM _hub_tmp")
            con.unregister("_hub_tmp")
    except Exception as _save_exc:
        print(f"[ML_INSIGHTS] Falha ao salvar hub no banco: {_save_exc}", flush=True)
    finally:
        try:
            con.close()
        except Exception:
            pass


def carregar_hub():
    df = pd.DataFrame()
    key = "id_cliente"

    con = _open_hub_db()
    if con is not None:
        try:
            _tbl = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [HUB_DB_TABLE],
            ).fetchone()[0]
            if _tbl:
                df = con.execute(f"SELECT * FROM {HUB_DB_TABLE}").df()

            _meta = con.execute("SELECT v FROM hub_meta WHERE k='hub_key' LIMIT 1").fetchone()
            if _meta and _meta[0]:
                key = str(_meta[0])
        except Exception:
            pass
        finally:
            try:
                con.close()
            except Exception:
                pass

    if not df.empty:
        return df, key

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



def require_authentication():
    if st.session_state.get("auth_ok", False):
        return

    _, col_c, _ = st.columns([1, 1.5, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; margin-bottom:24px;">
            <div style="display:inline-flex;align-items:center;justify-content:center;
                        width:64px;height:64px;background:rgba(78,140,255,0.12);
                        border-radius:16px;font-size:32px;margin-bottom:16px;">📊</div>
            <h2 style="font-size:24px;font-weight:700;margin-bottom:6px;color:#fafafa;">ML Insights Hub</h2>
            <p style="color:#a6a9b6;font-size:14px;margin:0;">
                Análise preditiva e automação inteligente para seu negócio</p>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("##### 🔑 Acesso ao painel")
            senha = st.text_input("Senha", type="password", placeholder="••••••••", label_visibility="collapsed")
            st.caption("Senha")
            if st.button("Acessar Dashboard", type="primary", width='stretch'):
                if hmac.compare_digest(senha, APP_PASSWORD):
                    st.session_state.auth_ok = True
                    st.rerun()
                else:
                    st.error("❌ Senha incorreta. Tente novamente.")
        st.caption("Senha padrão: mlhub123 | Personalize em Settings > Secrets no Streamlit Cloud.")
    st.stop()


try:
    require_authentication()
except Exception as _auth_err:
    st.error(f"Erro na autenticacao: {_auth_err}")
    st.code(_traceback.format_exc())
    st.stop()

# ── Cabecalho principal ───────────────────────────────────────
col_hdr, col_sair = st.columns([5, 1])
with col_hdr:
    st.markdown("# 📊 ML Insights Hub")
    st.markdown(
        "Importe planilhas em qualquer formato, consolide dados automaticamente "
        "e treine modelos de machine learning — tudo em um lugar."
    )
with col_sair:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Sair", width='stretch'):
        st.session_state.auth_ok = False
        st.rerun()
st.divider()


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
    # Converte tipo str nativo do Pandas 3.x para object antes de operar
    try:
        s = series.astype(object)
    except Exception:
        s = series
    cleaned = (
        s.astype(str)
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


def _detect_csv_sep(uploaded_file):
    uploaded_file.seek(0)
    sample = uploaded_file.read(8192)
    uploaded_file.seek(0)
    text_sample = sample.decode("utf-8", errors="ignore") if isinstance(sample, (bytes, bytearray)) else str(sample)
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ";" if text_sample.count(";") > text_sample.count(",") else ","


def read_uploaded_file(uploaded_file, nrows=None):
    name = uploaded_file.name.lower()
    if name.endswith(".parquet"):
        raw_bytes = uploaded_file.getvalue()
        df = pd.read_parquet(io.BytesIO(raw_bytes))
        if nrows and len(df) > nrows:
            df = df.head(nrows)
        return df
    if name.endswith(".csv"):
        sep = _detect_csv_sep(uploaded_file)
        df = pd.read_csv(uploaded_file, sep=sep, nrows=nrows, low_memory=False)
        uploaded_file.seek(0)
        return df
    # Para Excel, ainda precisa carregar o arquivo inteiro em memoria (openpyxl).
    raw_bytes = uploaded_file.getvalue()
    return pd.read_excel(io.BytesIO(raw_bytes), nrows=nrows)


def iter_uploaded_csv_chunks(uploaded_file, chunksize=100_000):
    sep = _detect_csv_sep(uploaded_file)
    try:
        for chunk in pd.read_csv(uploaded_file, sep=sep, chunksize=chunksize, low_memory=False):
            yield chunk
    finally:
        uploaded_file.seek(0)


def compute_upload_hash(files):
    if not files:
        return ""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name.lower()):
        h.update(f.name.encode("utf-8", errors="ignore"))
        h.update(str(getattr(f, "size", 0)).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _hashable_cols(df):
    """Retorna apenas colunas cujos valores são hashable (seguro para drop_duplicates sem subset)."""
    safe = []
    for c in df.columns:
        try:
            df[c].drop_duplicates()
            safe.append(c)
        except TypeError:
            pass
    return safe


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


def merge_clean_into_hub(hub, clean, preferred_key):
    current_key = preferred_key if preferred_key in clean.columns else suggest_key_column(clean)

    if current_key in clean.columns:
        clean = clean.dropna(subset=[current_key])
        valid_count = int(clean[current_key].notna().sum())
        uniq_count = int(clean[current_key].nunique(dropna=True))
        uniq_ratio = (uniq_count / max(1, valid_count)) if valid_count > 0 else 0
        if valid_count > 0 and uniq_ratio >= 0.95:
            clean = clean.drop_duplicates(subset=[current_key], keep="last")
            hub = upsert_hub(hub, clean, current_key)
            return hub, current_key

    hub = pd.concat([hub, clean], ignore_index=True)
    _safe = _hashable_cols(hub)
    if _safe:
        hub = hub.drop_duplicates(subset=_safe, keep="last")
    return hub, None


def merge_clean_into_hub_sampled(hub, clean, preferred_key, cap_rows):
    current_key = preferred_key if preferred_key in clean.columns else suggest_key_column(clean)

    if current_key in clean.columns:
        clean = clean.dropna(subset=[current_key])
        clean = clean.drop_duplicates(subset=[current_key], keep="last")

    if hub is None or hub.empty:
        merged = clean.copy()
    else:
        all_cols = sorted(set(hub.columns).union(set(clean.columns)))
        merged = pd.concat(
            [hub.reindex(columns=all_cols), clean.reindex(columns=all_cols)],
            ignore_index=True,
        )
        if current_key in merged.columns:
            merged = merged.drop_duplicates(subset=[current_key], keep="last")
        else:
            _safe_m = _hashable_cols(merged)
            if _safe_m:
                merged = merged.drop_duplicates(subset=_safe_m, keep="last")

    if len(merged) > cap_rows:
        merged = merged.sample(cap_rows, random_state=42)

    return merged, (current_key if current_key in clean.columns else None)


def suggest_target(df):
    if df is None or len(df.columns) == 0:
        return "target"
    for c in ["contratou", "target", "y"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def suggest_key_column(df):
    if df is None or df.empty:
        return "id_cliente"

    cols = df.columns.tolist()
    if not cols:
        return "id_cliente"
    id_like = [
        c for c in cols
        if ("id" in c.lower()) or c.lower().endswith("_id") or ("codigo" in c.lower())
    ]
    ranked = id_like + [c for c in cols if c not in id_like]

    best_col = ranked[0]
    best_score = -1.0
    best_unique = -1

    for c in ranked:
        s = df[c]
        non_null = int(s.notna().sum())
        if non_null == 0:
            continue
        uniq = int(s.nunique(dropna=True))
        ratio = uniq / max(1, non_null)
        score = ratio + (0.05 if c in id_like else 0.0)

        if score > best_score or (score == best_score and uniq > best_unique):
            best_col = c
            best_score = score
            best_unique = uniq

    return best_col


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


# ── Geocodificacao e Mapa ─────────────────────────────────────────────────────
def detect_geo_columns(df):
    """Detecta colunas de lat/lon, CEP, rua e dados de endereco no dataframe."""
    cl = {c: normalize_name(c) for c in df.columns}
    lat_col = next((c for c, l in cl.items() if l in {"lat", "latitude"}), None)
    lon_col = next((c for c, l in cl.items() if l in {"lon", "lng", "longitude", "long"}), None)
    cep_col = next((c for c, l in cl.items() if l in {"cep", "cod_postal", "codigo_postal", "zip", "zip_code"}
                    or "cep" in l), None)
    rua_col = next((c for c, l in cl.items() if l in {"rua", "logradouro", "endereco", "enderezo", "street", "address"}), None)
    nome_col = next((c for c, l in cl.items() if l in {"nome", "nome_cliente", "razao_social", "cliente", "name"}), None)
    bairro_col = next((c for c, l in cl.items() if l in {"bairro", "neighborhood", "distrito"}), None)
    cidade_col = next((c for c, l in cl.items() if l in {"cidade", "municipio", "city"}), None)
    estado_col = next((c for c, l in cl.items() if l in {"estado", "uf", "state"}), None)
    return lat_col, lon_col, cep_col, rua_col, nome_col, bairro_col, cidade_col, estado_col


@st.cache_data(show_spinner=False, ttl=21600)
def geocode_ceps(ceps_tuple: tuple) -> dict:
    """Geocodifica CEPs via BrasilAPI. Retorna {cep_8dig: {lat, lon, rua, bairro, cidade, estado}}."""
    resultado = {}
    for cep in ceps_tuple:
        cep_clean = re.sub(r"[^0-9]", "", str(cep))[:8]
        if len(cep_clean) != 8:
            continue
        try:
            resp = requests.get(
                f"https://brasilapi.com.br/api/cep/v2/{cep_clean}",
                timeout=6,
                headers={"User-Agent": "MLInsightsHub/1.0"},
            )
            if resp.ok:
                d = resp.json()
                loc = d.get("location") or {}
                coords = loc.get("coordinates") or {}
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                if lat and lon:
                    resultado[cep_clean] = {
                        "lat": float(lat),
                        "lon": float(lon),
                        "rua": str(d.get("street") or ""),
                        "bairro": str(d.get("neighborhood") or ""),
                        "cidade": str(d.get("city") or ""),
                        "estado": str(d.get("state") or ""),
                    }
        except Exception:
            pass
    return resultado


def _render_map(map_df: pd.DataFrame, cat_cols_avail: list):
    """Renderiza mapa interativo ScatterplotLayer (pydeck/CARTO) ou fallback st.map."""
    MAX_MAP_ROWS = 3_000
    if len(map_df) > MAX_MAP_ROWS:
        st.warning(f"Exibindo amostra de {MAX_MAP_ROWS:,} de {len(map_df):,} pontos para performance.")
        map_df = map_df.sample(MAX_MAP_ROWS, random_state=42).reset_index(drop=True)
    else:
        map_df = map_df.reset_index(drop=True)

    # Colorir por categoria
    color_col = None
    _avail = [c for c in cat_cols_avail if c in map_df.columns]
    if _avail:
        _color_sel = st.selectbox("Colorir pontos por", ["(nenhum)"] + _avail, key="map_color")
        if _color_sel != "(nenhum)":
            color_col = _color_sel

    _palette = [
        (78, 140, 255), (244, 162, 97), (46, 204, 113), (231, 76, 60),
        (155, 89, 182), (26, 188, 156), (243, 156, 18), (52, 152, 219),
    ]

    map_df = map_df.copy()
    if color_col and color_col in map_df.columns:
        _cats = map_df[color_col].astype(str).unique().tolist()[:8]
        _cmap = {c: _palette[i % len(_palette)] for i, c in enumerate(_cats)}
        _colors = [_cmap.get(str(v), (128, 128, 128)) for v in map_df[color_col]]
        st.markdown("**Legenda:** " + " \u2502 ".join(f"\u25cf {c}" for c in _cats))
    else:
        _colors = [(78, 140, 255)] * len(map_df)

    map_df["__r"] = [c[0] for c in _colors]
    map_df["__g"] = [c[1] for c in _colors]
    map_df["__b"] = [c[2] for c in _colors]

    _tip_fields = [c for c in ["nome", "rua", "bairro", "cidade", "estado"] if c in map_df.columns]
    if color_col and color_col in map_df.columns and color_col not in _tip_fields:
        _tip_fields = [color_col] + _tip_fields
    _tip_html = "<br>".join(
        f"<b>{f.replace('_',' ').title()}:</b> {{{f}}}" for f in _tip_fields
    ) or "<b>Lat:</b> {lat}<br><b>Lon:</b> {lon}"

    if _PYDECK_OK:
        try:
            _layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_color=["__r", "__g", "__b", 210],
                get_radius=120,
                pickable=True,
                auto_highlight=True,
            )
            _view = pdk.ViewState(
                latitude=float(map_df["lat"].mean()),
                longitude=float(map_df["lon"].mean()),
                zoom=11, pitch=0,
            )
            _deck = pdk.Deck(
                layers=[_layer],
                initial_view_state=_view,
                tooltip={
                    "html": _tip_html,
                    "style": {"color": "white", "backgroundColor": "#1e2530", "padding": "8px"},
                },
                map_provider="carto",
                map_style="dark",
            )
            st.pydeck_chart(_deck, width='stretch')
        except Exception as _pdk_err:
            st.warning(f"pydeck indisponivel ({_pdk_err}). Usando mapa simples.")
            st.map(map_df[["lat", "lon"]])
    else:
        st.map(map_df[["lat", "lon"]])

    _show = [c for c in ["nome", "rua", "bairro", "cidade", "estado", "lat", "lon"] if c in map_df.columns]
    if _show:
        with st.expander("📋 Ver tabela de endereços plotados"):
            st.dataframe(map_df[_show].reset_index(drop=True), width='stretch')


# ── Estado inicial ───────────────────────────────────────────
if "confirm_limpar" not in st.session_state:
    st.session_state.confirm_limpar = False
if "upload_attempted" not in st.session_state:
    st.session_state.upload_attempted = False
if "auto_run_ml" not in st.session_state:
    st.session_state.auto_run_ml = False
if "last_processed_hash" not in st.session_state:
    st.session_state.last_processed_hash = ""
if "last_update_at" not in st.session_state:
    st.session_state.last_update_at = "-"
if "hub_total_rows" not in st.session_state:
    st.session_state.hub_total_rows = 0
if "large_mode_active" not in st.session_state:
    st.session_state.large_mode_active = False
if st.session_state.get("app_schema_version") != APP_SCHEMA_VERSION:
    if HUB_FILE.exists():
        HUB_FILE.unlink()
    if HUB_KEY_FILE.exists():
        HUB_KEY_FILE.unlink()
    st.session_state.hub_df = pd.DataFrame()
    st.session_state.hub_key = "id_cliente"
    st.session_state.upload_attempted = False
    st.session_state.last_processed_hash = ""
    st.session_state.last_update_at = "-"
    st.session_state.hub_total_rows = 0
    st.session_state.large_mode_active = False
    st.session_state.app_schema_version = APP_SCHEMA_VERSION

# ── Carrega dados persistidos do disco quando a sessão começa ─
if "hub_df" not in st.session_state:
    _disk_hub, _disk_key = carregar_hub()
    st.session_state.hub_df = _disk_hub
    st.session_state.hub_key = st.session_state.get("hub_key", _disk_key)
    st.session_state.hub_total_rows = len(_disk_hub)
    if not _disk_hub.empty:
        st.session_state.last_update_at = "recuperado do disco"

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 📊 ML Insights Hub")
st.sidebar.caption(f"Build: {APP_SCHEMA_VERSION}")
if IS_CLOUD and str(HUB_DB_PATH) == "dados.duckdb":
    st.sidebar.warning(
        "Persistência limitada: configure HUB_DB_PATH (ex: md:seu_db) para manter dados após reinícios do app.",
        icon="⚠️",
    )
else:
    st.sidebar.caption(f"Banco conectado: {HUB_DB_PATH}")
st.sidebar.markdown("---")

# ── Upload ────────────────────────────────────────────────────
st.sidebar.markdown("### 📁 Sua planilha")
st.sidebar.info(
    "**Arquivo grande (>100 MB)?**\n\n"
    "1. Baixe `preparar_dados.py` do GitHub\n"
    "2. Execute: `uv run preparar_dados.py seu_arquivo.xlsx`\n"
    "3. Envie o `.parquet` gerado aqui\n\n"
    "Parquet é 5–10× menor e carrega sem travar.",
    icon="💡",
)
uploaded_files = st.sidebar.file_uploader(
    "Parquet / CSV / Excel",
    type=["parquet", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.sidebar.caption("Prioridade: .parquet (rápido) › .csv › .xlsx")
current_upload_hash = compute_upload_hash(uploaded_files)
total_upload_mb = (sum(getattr(f, "size", 0) for f in uploaded_files) / (1024 * 1024)) if uploaded_files else 0
large_mode = total_upload_mb >= LARGE_FILE_THRESHOLD_MB
if large_mode:
    st.sidebar.warning(
        f"Arquivo grande detectado ({total_upload_mb:.1f} MB). "
        "Modo robusto por chunks ativado."
    )

# ── Config automática baseada no primeiro arquivo ou hub ──────
_has_hub = not st.session_state.hub_df.empty

if uploaded_files:
    try:
        _is_parquet_prev = uploaded_files[0].name.lower().endswith(".parquet")
        _preview_rows = None if _is_parquet_prev else (5000 if large_mode else None)
        _raw_prev = read_uploaded_file(uploaded_files[0], nrows=_preview_rows)
        _df_cfg, _, _ = preprocess_df(_raw_prev)
    except Exception as _prev_err:
        print(f"[ML_INSIGHTS] preview error: {_prev_err}", flush=True)
        _df_cfg = st.session_state.hub_df.copy()
elif _has_hub:
    _df_cfg = st.session_state.hub_df.copy()
else:
    _df_cfg = pd.DataFrame()

_num_cfg = _df_cfg.select_dtypes(include=np.number).columns.tolist()
_all_cfg = _df_cfg.columns.tolist()
_auto_target = suggest_target(_df_cfg)
_auto_key = suggest_key_column(_df_cfg)
_cfg_has_columns = len(_all_cfg) > 0

def _detect_tipo(df_d, tgt):
    if tgt not in df_d.columns:
        return "Clustering (agrupamento)"
    _col = df_d[tgt].dropna()
    if _col.nunique() <= 2:
        return "Classificacao (sim/nao)"
    if pd.api.types.is_numeric_dtype(_col) and _col.nunique() > 10:
        return "Regressao (valor numerico)"
    if _col.nunique() <= 10:
        return "Classificacao (sim/nao)"
    return "Regressao (valor numerico)"

_auto_tipo_str = _detect_tipo(_df_cfg, _auto_target)
_tipo_opts = ["Classificacao (sim/nao)", "Regressao (valor numerico)", "Clustering (agrupamento)"]

# ── JOIN de planilhas (visivel quando 2+ arquivos enviados) ───
if uploaded_files and len(uploaded_files) >= 2:
    with st.sidebar.expander("🔗 Cruzar planilhas (JOIN)", expanded=False):
        st.caption(
            "Una dados de duas planilhas pela coluna em comum, "
            "mesmo que as colunas tenham nomes diferentes nas duas."
        )
        _jnames = [f.name for f in uploaded_files]
        _ja_name = st.selectbox("Planilha base (A)", _jnames, index=0, key="join_fa")
        _jb_name = st.selectbox("Planilha para juntar (B)", _jnames,
                                 index=min(1, len(_jnames) - 1), key="join_fb")

        if _ja_name != _jb_name:
            try:
                _fa = next(f for f in uploaded_files if f.name == _ja_name)
                _fb = next(f for f in uploaded_files if f.name == _jb_name)
                _prev_a, _, _ = preprocess_df(read_uploaded_file(_fa, nrows=500))
                _prev_b, _, _ = preprocess_df(read_uploaded_file(_fb, nrows=500))
                _cols_a = _prev_a.columns.tolist()
                _cols_b = _prev_b.columns.tolist()

                _jca = st.selectbox("Chave em A", _cols_a, key="join_ka")
                _jcb = st.selectbox(
                    "Chave em B",
                    _cols_b,
                    index=(_cols_b.index(_jca) if _jca in _cols_b else 0),
                    key="join_kb",
                )
                _jtype = st.radio(
                    "Tipo de JOIN",
                    ["left", "inner", "outer"],
                    format_func=lambda x: {
                        "left": "LEFT — mantém todos de A",
                        "inner": "INNER — só registros comuns",
                        "outer": "OUTER — todos os registros",
                    }[x],
                    key="join_type",
                )
                _cols_excl = st.multiselect(
                    "Colunas de B para ignorar (evitar duplicatas)",
                    [c for c in _cols_b if c != _jcb],
                    key="join_excl",
                )

                # Previa do resultado
                if st.checkbox("Previa do resultado (5 linhas)", key="join_prev"):
                    if _jca != _jcb:
                        _prev_b2 = _prev_b.rename(columns={_jcb: _jca})
                    else:
                        _prev_b2 = _prev_b.copy()
                    _excl_m = [canonical_column(c) for c in _cols_excl]
                    _prev_b2 = _prev_b2.drop(columns=_excl_m, errors="ignore")
                    _ov = [c for c in _prev_b2.columns if c in _prev_a.columns and c != _jca]
                    _prev_b2 = _prev_b2.drop(columns=_ov, errors="ignore")
                    _prev_join = _prev_a.merge(_prev_b2, on=_jca, how=_jtype)
                    st.dataframe(_prev_join.head(5), width='stretch')

                if st.button("🔗 Aplicar JOIN no Hub", key="do_join", width='stretch'):
                    with st.spinner("Aplicando JOIN..."):
                        _fa.seek(0)
                        _fb.seek(0)
                        _full_a, _, _ = preprocess_df(read_uploaded_file(_fa))
                        _full_b, _, _ = preprocess_df(read_uploaded_file(_fb))
                        if _jca != _jcb:
                            _full_b = _full_b.rename(columns={_jcb: _jca})
                        _excl_mapped = [canonical_column(c) for c in _cols_excl]
                        _full_b = _full_b.drop(columns=_excl_mapped, errors="ignore")
                        _overlap = [c for c in _full_b.columns if c in _full_a.columns and c != _jca]
                        _full_b = _full_b.drop(columns=_overlap, errors="ignore")
                        _joined = _full_a.merge(_full_b, on=_jca, how=_jtype)
                        if len(_joined) > MAX_DASHBOARD_ROWS:
                            _joined = _joined.sample(MAX_DASHBOARD_ROWS, random_state=42).reset_index(drop=True)
                        st.session_state.hub_df = _joined
                        st.session_state.hub_key = _jca
                        st.session_state.hub_total_rows = len(_joined)
                        st.session_state.last_update_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        salvar_hub(_joined, _jca)
                    st.success(f"✅ JOIN aplicado: {len(_joined):,} registros.")
                    st.rerun()
            except Exception as _je:
                st.error(f"Erro no JOIN: {_je}")
        else:
            st.warning("Selecione planilhas diferentes para fazer o JOIN.")

# ── Configurações avançadas (expander) ───────────────────────
with st.sidebar.expander("⚙️ Configurações avançadas (opcional)"):
    if _cfg_has_columns:
        target_col = st.selectbox(
            "Coluna alvo (o que prever)",
            options=_all_cfg,
            index=_all_cfg.index(_auto_target) if _auto_target in _all_cfg else 0,
            help="Ex: contratou, valor, segmento.",
        )
        _feats_opts = [c for c in _num_cfg if c != target_col]
        features_selecionadas = st.multiselect(
            "Variáveis preditoras",
            options=_feats_opts,
            default=_feats_opts,
            help="Colunas numéricas usadas no modelo.",
        )
        tipo_problema = st.radio(
            "Tipo de análise",
            _tipo_opts,
            index=_tipo_opts.index(_auto_tipo_str),
        )
        key_col = st.selectbox(
            "Chave única (evita duplicatas)",
            options=_all_cfg,
            index=_all_cfg.index(_auto_key) if _auto_key in _all_cfg else 0,
        )
    else:
        st.warning("Não foi possível detectar colunas válidas na prévia do arquivo.")
        target_col = "target"
        features_selecionadas = []
        tipo_problema = "Clustering (agrupamento)"
        key_col = "id_cliente"

st.sidebar.markdown("---")

# ── Botão único ───────────────────────────────────────────────
_can_analyze = (bool(uploaded_files) or _has_hub) and _cfg_has_columns
rodar = st.session_state.pop("auto_run_ml", False)

_click_atualizar = st.sidebar.button(
    "▶ Atualizar Dashboard",
    type="primary",
    width='stretch',
    disabled=not _can_analyze,
)
if not _can_analyze:
    if uploaded_files and not _cfg_has_columns:
        st.sidebar.caption("⚠️ Arquivo sem estrutura válida. Verifique cabeçalhos e formatação.")
    else:
        st.sidebar.caption("⬆️ Arraste sua planilha acima para começar")
elif _has_hub and not uploaded_files:
    st.sidebar.caption(f"Hub ativo: {len(st.session_state.hub_df):,} registros")

# ── Auto-disparo: processa automaticamente ao detectar arquivo novo ──
_novo_upload = (
    bool(uploaded_files)
    and bool(current_upload_hash)
    and current_upload_hash != st.session_state.get("last_processed_hash", "")
)
_auto_processar_upload = _novo_upload and not _click_atualizar
analisar_tudo = _click_atualizar or _auto_processar_upload

# ── Lógica do botão ───────────────────────────────────────────
if analisar_tudo and uploaded_files:
    st.session_state.upload_attempted = True
    _upload_error_main = None
    with st.spinner("⏳ Carregando e processando dados..."):
      try:
        hub = st.session_state.hub_df.copy()
        rows_ingested = 0
        base_total_rows = int(st.session_state.get("hub_total_rows") or len(hub))
        processed_count = 0
        ignored_count = 0
        for f in uploaded_files:
            try:
                # ── Parquet: caminho mais eficiente, sem chunking necessário ──
                if f.name.lower().endswith(".parquet"):
                    _pq_bytes = f.getvalue()
                    raw = pd.read_parquet(io.BytesIO(_pq_bytes))
                    del _pq_bytes
                    if len(raw) > MAX_DASHBOARD_ROWS:
                        raw = raw.sample(MAX_DASHBOARD_ROWS, random_state=42).reset_index(drop=True)
                    clean, _, _ = preprocess_df(raw)
                    del raw
                    if clean.empty:
                        ignored_count += 1
                        continue
                    rows_ingested += len(clean)
                    hub, used_key = merge_clean_into_hub_sampled(hub, clean, key_col, MAX_DASHBOARD_ROWS)
                    if used_key:
                        st.session_state.hub_key = used_key
                    processed_count += 1
                    continue

                if large_mode and f.name.lower().endswith(".csv"):
                    file_had_rows = False
                    for chunk in iter_uploaded_csv_chunks(f):
                        clean, _, _ = preprocess_df(chunk)
                        if clean.empty:
                            continue
                        file_had_rows = True
                        rows_ingested += len(clean)
                        hub, used_key = merge_clean_into_hub_sampled(
                            hub,
                            clean,
                            key_col,
                            MAX_DASHBOARD_ROWS,
                        )
                        if used_key:
                            st.session_state.hub_key = used_key
                    if file_had_rows:
                        processed_count += 1
                    else:
                        ignored_count += 1
                    continue
                if large_mode and f.name.lower().endswith((".xlsx", ".xls")):
                    # Converte Excel grande automaticamente chunk a chunk via pandas
                    st.sidebar.info(f"🔄 {f.name}: Excel grande — convertendo automaticamente...")
                    try:
                        _xl_bytes = io.BytesIO(f.getvalue())
                        _xl_all = pd.read_excel(_xl_bytes, dtype=str)
                        _xl_bytes.close()
                        file_had_rows = False
                        for _start in range(0, len(_xl_all), 50_000):
                            chunk = _xl_all.iloc[_start:_start + 50_000].copy()
                            clean, _, _ = preprocess_df(chunk)
                            if clean.empty:
                                continue
                            file_had_rows = True
                            rows_ingested += len(clean)
                            hub, used_key = merge_clean_into_hub_sampled(
                                hub, clean, key_col, MAX_DASHBOARD_ROWS
                            )
                            if used_key:
                                st.session_state.hub_key = used_key
                        del _xl_all
                        if file_had_rows:
                            processed_count += 1
                        else:
                            ignored_count += 1
                    except Exception as _xl_err:
                        st.sidebar.error(f"❌ {f.name}: falha ao converter Excel — {_xl_err}")
                    continue
                else:
                    raw = read_uploaded_file(f)

                clean, _, _ = preprocess_df(raw)
                if clean.empty:
                    ignored_count += 1
                    continue

                rows_ingested += len(clean)
                hub, used_key = merge_clean_into_hub(hub, clean, key_col)
                if used_key:
                    st.session_state.hub_key = used_key

                processed_count += 1
            except MemoryError:
                st.sidebar.error(
                    f"❌ {f.name}: arquivo muito grande para a memória disponível. "
                    "Converta para CSV e tente novamente."
                )
            except Exception as _e:
                st.sidebar.error(f"❌ Erro ao processar {f.name}: {_e}")
                import traceback as _tb
                print(f"[ML_INSIGHTS] Erro em {f.name}:", _tb.format_exc(), flush=True)

        rows_after = len(hub)
        st.session_state.hub_df = hub
        if large_mode:
            st.session_state.hub_total_rows = base_total_rows + rows_ingested
        else:
            st.session_state.hub_total_rows = rows_after
        st.session_state.large_mode_active = large_mode
        st.session_state.last_processed_hash = current_upload_hash
        st.session_state.last_update_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        salvar_hub(hub, st.session_state.get("hub_key", key_col))
      except Exception as _upload_main_exc:
        _upload_error_main = _upload_main_exc
        import traceback as _tb2
        print(f"[ML_INSIGHTS] Erro crítico no upload:", _tb2.format_exc(), flush=True)
        st.session_state.last_processed_hash = current_upload_hash  # evita loop

    if _upload_error_main is not None:
        st.error(f"❌ Erro ao processar o arquivo: `{_upload_error_main}`\n\nDetalhes no log do app.")
    elif processed_count == 0 or rows_after == 0:
        st.sidebar.error(
            "Nenhum registro válido foi consolidado. "
            "Abra as Configurações avançadas e ajuste a chave única."
        )
        rodar = False
    else:
        st.session_state.auto_run_ml = _click_atualizar
        if large_mode:
            st.sidebar.success(
                f"✅ Dashboard atualizado. Total processado: {st.session_state.hub_total_rows:,} | "
                f"Amostra em memória: {rows_after:,}."
            )
        else:
            st.sidebar.success(f"✅ Dashboard atualizado com {rows_after:,} registros.")
        if _auto_processar_upload:
            st.sidebar.caption("Dados carregados. Clique em 'Atualizar Dashboard' quando quiser rodar o Machine Learning.")
        st.rerun()
elif analisar_tudo and _has_hub:
    rodar = True

if st.sidebar.button("🗑 Limpar Hub", width='stretch'):
    st.session_state.confirm_limpar = True

if st.session_state.confirm_limpar:
    st.sidebar.warning("⚠️ Isso apagará **todos os dados** do hub. Confirma?")
    _cy, _cn = st.sidebar.columns(2)
    if _cy.button("Sim, limpar", width='stretch', key="confirm_yes"):
        st.session_state.hub_df = pd.DataFrame()
        if HUB_FILE.exists():
            HUB_FILE.unlink()
        if HUB_KEY_FILE.exists():
            HUB_KEY_FILE.unlink()
        _db_con = _open_hub_db()
        if _db_con is not None:
            try:
                _db_con.execute(f"DROP TABLE IF EXISTS {HUB_DB_TABLE}")
                _db_con.execute("DELETE FROM hub_meta WHERE k='hub_key'")
            except Exception:
                pass
            finally:
                try:
                    _db_con.close()
                except Exception:
                    pass
        st.session_state.confirm_limpar = False
        st.rerun()
    if _cn.button("Cancelar", width='stretch', key="confirm_no"):
        st.session_state.confirm_limpar = False
        st.rerun()



# ── Estado do hub ─────────────────────────────────────────────
df = st.session_state.hub_df.copy()

# ── Banner estado vazio (sem dados de demo) ──────────────────
if df.empty:
    st.info("📂 Nenhum dado carregado. Arraste sua planilha na barra lateral para iniciar a análise.")
    if st.session_state.get("last_update_at", "-") != "-":
        st.caption(f"Último processamento: {st.session_state.last_update_at}")
    st.stop()

# Em bases muito grandes, usa uma amostra para visualizacao e calculos de dashboard.
if len(df) > MAX_DASHBOARD_ROWS:
    df_view = df.sample(MAX_DASHBOARD_ROWS, random_state=42)
    st.warning(
        f"Base muito grande ({len(df):,} linhas). "
        f"Dashboard exibindo amostra de {MAX_DASHBOARD_ROWS:,} linhas para estabilidade."
    )
else:
    df_view = df

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
numeric_cols = df_view.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df_view.columns if c not in numeric_cols and df_view[c].nunique() <= 30]

# Limitar colunas para evitar graficos pesados demais
MAX_CHART_NUM = 12
MAX_CHART_CAT = 4
MAX_CORR_COLS = 10
MAX_SCATTER_ROWS = 5_000
numeric_cols_chart = numeric_cols[:MAX_CHART_NUM]
cat_cols_chart = cat_cols[:MAX_CHART_CAT]
numeric_cols_corr = numeric_cols[:MAX_CORR_COLS]

st.markdown("## 📈 Dashboard Executivo")
st.caption(
    "Visao completa dos dados consolidados no hub. "
    "Use as abas abaixo para navegar entre os paineis."
)

missing_total = int(df_view.isnull().sum().sum())
missing_pct = (missing_total / (max(1, df_view.shape[0] * df_view.shape[1]))) * 100

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
_total_display = st.session_state.get("hub_total_rows") or df.shape[0]
kpi_1.metric("👥 Clientes Analisados", format_number(_total_display, abbreviate=True))
kpi_2.metric("📊 Variáveis de Análise", f"{df.shape[1]}")
kpi_3.metric("⚠️ Dados Incompletos", format_number(missing_total, abbreviate=True))
kpi_4.metric("✅ Qualidade dos Dados", f"{max(0, 100 - missing_pct):.1f}%")

tab_overview, tab_profile, tab_rel, tab_mapa, tab_export = st.tabs([
    "📋 Visao Geral",
    "📊 Perfil dos Dados",
    "🔗 Relacoes e Tendencias",
    "🗺️ Mapa de Clientes",
    "💾 Exportacao",
])

with tab_overview:
    st.markdown("Resumo rapido dos dados carregados no hub. Confira os KPIs, a amostra e as estatisticas principais.")
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Amostra dos dados")
        st.dataframe(df_view.head(15), width='stretch')
    with right:
        st.markdown("#### Estatisticas descritivas")
        if numeric_cols:
            summary = df_view[numeric_cols_chart].describe().T[["mean", "std", "min", "max"]].round(2)
            st.dataframe(summary, width='stretch')
        else:
            st.info("Nenhuma coluna numerica encontrada para estatisticas.")

    if numeric_cols:
        st.markdown("#### Indicadores por variável")
        kpi_cols = numeric_cols[:6]
        kpi_grid = st.columns(len(kpi_cols))
        for i, c in enumerate(kpi_cols):
            total = df_view[c].sum()
            media = df_view[c].mean()
            mediana = df_view[c].median()
            kpi_grid[i].metric(
                label=c.replace("_", " ").title(),
                value=format_number(media, decimals=1, abbreviate=True),
                delta=f"total {format_number(total, abbreviate=True)} | med {format_number(mediana, abbreviate=True)}",
            )

with tab_profile:
    st.markdown("Visualize como os dados se distribuem — ideal para identificar padroes, outliers e concentracoes.")
    if numeric_cols:
        st.markdown("#### Distribuição das variáveis numéricas")
        n = len(numeric_cols_chart)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        
        _fig_dist = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[c.replace("_", " ").title() for c in numeric_cols_chart],
            specs=[[{"secondary_y": False}] * ncols for _ in range(nrows)],
        )
        
        for i, c in enumerate(numeric_cols_chart):
            row = (i // ncols) + 1
            col = (i % ncols) + 1
            _data = df_view[c].dropna().values
            _fig_dist.add_trace(
                go.Histogram(x=_data, nbinsx=25, name=c.replace("_", " "), 
                            marker=dict(color="#4e8cff", opacity=0.8),
                            hovertemplate="<b>%{x:.2f}</b><br>Frequência: %{y}<extra></extra>"),
                row=row, col=col
            )
        
        _fig_dist.update_xaxes(title_text="Valor", row=nrows, col=1)
        _fig_dist.update_yaxes(title_text="Frequência", row=1, col=1)
        _fig_dist.update_layout(height=350 * nrows, showlegend=False, hovermode="x unified")
        st.plotly_chart(_fig_dist, width='stretch')
        del _fig_dist
        gc.collect()

    if cat_cols:
        st.markdown("#### Distribuição das categorias")
        for c in cat_cols_chart:
            vcounts = df_view[c].value_counts().head(10).reset_index()
            vcounts.columns = [c, "Quantidade"]
            _fig_cat = px.bar(
                vcounts, 
                x="Quantidade", 
                y=c,
                orientation="h",
                title=f"Top 10 - {c.replace('_', ' ').title()}",
                color="Quantidade",
                color_continuous_scale="Viridis",
                hover_data={"Quantidade": ":,"},
                height=400
            )
            _fig_cat.update_layout(yaxis={"categoryorder": "total ascending"}, hovermode="closest")
            st.plotly_chart(_fig_cat, width='stretch')
            del _fig_cat
        gc.collect()

with tab_rel:
    st.markdown("Explore relacoes entre variaveis. Correlacoes altas indicam que uma variavel pode prever a outra.")
    if len(numeric_cols_corr) >= 2:
        st.markdown("#### Mapa de correlação")
        st.caption("Valores próximos de +1 ou -1 indicam relação forte. Próximos de 0 indicam independência.")
        corr = df_view[numeric_cols_corr].corr(numeric_only=True)
        
        _fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=[c.replace("_", " ") for c in corr.columns],
            y=[c.replace("_", " ") for c in corr.columns],
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 9},
            colorbar=dict(title="Correlação"),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlação: %{z:.3f}<extra></extra>",
        ))
        _fig_corr.update_layout(height=600, xaxis={"side": "bottom"}, hovermode="closest")
        st.plotly_chart(_fig_corr, width='stretch')
        del _fig_corr, corr
        gc.collect()

        st.markdown("#### Relação entre variáveis (Scatter Interativo)")
        st.caption("Selecione dois ou três atributos para visualização interativa. Passe o mouse sobre os pontos para mais detalhes.")
        rel1, rel2, rel3, rel4 = st.columns(4)
        col_x = rel1.selectbox("Eixo X", numeric_cols_corr, index=0, key="dash_x")
        col_y = rel2.selectbox("Eixo Y", numeric_cols_corr, index=min(1, len(numeric_cols_corr) - 1), key="dash_y")
        col_z = rel3.selectbox("Eixo Z (3D)", ["Nenhum"] + numeric_cols_corr, key="dash_z")
        color_by = rel4.selectbox("Colorir por", ["Nenhum"] + cat_cols_chart, key="dash_color")

        _sc_df = df_view if len(df_view) <= MAX_SCATTER_ROWS else df_view.sample(MAX_SCATTER_ROWS, random_state=42)
        _sc_df = _sc_df.dropna(subset=[col_x, col_y] + ([col_z] if col_z != "Nenhum" else []))
        
        if col_z != "Nenhum" and len(_sc_df) > 0:
            _fig_sc = px.scatter_3d(
                _sc_df,
                x=col_x, y=col_y, z=col_z,
                color=color_by if color_by != "Nenhum" else None,
                hover_data={col_x: ":.2f", col_y: ":.2f", col_z: ":.2f"},
                title="Scatter Plot 3D Interativo",
                height=700
            )
        else:
            _fig_sc = px.scatter(
                _sc_df,
                x=col_x, y=col_y,
                color=color_by if color_by != "Nenhum" else None,
                hover_data={col_x: ":.2f", col_y: ":.2f"},
                title="Scatter Plot Interativo",
                height=600
            )
        
        _fig_sc.update_layout(hovermode="closest")
        st.plotly_chart(_fig_sc, width='stretch')
        del _fig_sc
        gc.collect()

    date_cols = [c for c in df_view.columns if "data" in c.lower() or "date" in c.lower() or "mes" in c.lower()]
    if date_cols and numeric_cols_chart:
        st.markdown("### Evolução temporal")
        dcol = date_cols[0]
        vcol = st.selectbox("Variável temporal", numeric_cols_chart, key="ts_var")
        try:
            df_ts = df_view[[dcol, vcol]].dropna().copy()
            df_ts[dcol] = pd.to_datetime(df_ts[dcol], dayfirst=False, errors="coerce")
            df_ts = df_ts.dropna(subset=[dcol]).sort_values(dcol)
            df_ts_g = df_ts.groupby(dcol)[vcol].mean()
            fig_ts, ax_ts = plt.subplots(figsize=(9, 4))
            ax_ts.plot(df_ts_g.index, df_ts_g.values, color="#e74c3c", linewidth=2)
            ax_ts.fill_between(df_ts_g.index, df_ts_g.values, alpha=0.15, color="#e74c3c")
            ax_ts.set_xlabel("Data")
            ax_ts.set_ylabel(vcol.replace("_", " ").title())
            fig_ts.tight_layout()
            st.pyplot(fig_ts)
            plt.close(fig_ts)
            gc.collect()
        except Exception:
            st.info("Coluna de data encontrada, mas não foi possível interpretar os valores.")

with tab_mapa:
    st.markdown("#### 🗺️ Mapa de Clientes por Endereço")
    st.caption(
        "Localiza cada cliente como um ponto no mapa até o nível de rua. "
        "O app detecta automaticamente colunas de lat/lon, CEP ou permite configuração manual."
    )

    _lat_c, _lon_c, _cep_c, _rua_c, _nome_c, _bairro_c, _cidade_c, _estado_c = detect_geo_columns(df_view)

    def _build_point_df(base_df, lat_col, lon_col):
        m = pd.DataFrame()
        m["lat"] = pd.to_numeric(base_df[lat_col], errors="coerce")
        m["lon"] = pd.to_numeric(base_df[lon_col], errors="coerce")
        for col_name, col_src in [
            ("nome", _nome_c), ("rua", _rua_c), ("bairro", _bairro_c),
            ("cidade", _cidade_c), ("estado", _estado_c),
        ]:
            if col_src and col_src in base_df.columns:
                m[col_name] = base_df[col_src].astype(str).values
        return m.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # ── Estratégia 1: colunas lat/lon detectadas automaticamente ──
    if _lat_c and _lon_c:
        st.success(f"✅ Coordenadas detectadas: **{_lat_c}** / **{_lon_c}**")
        _map_df = _build_point_df(df_view, _lat_c, _lon_c)
        if not _map_df.empty:
            st.markdown(f"**{len(_map_df):,} clientes** com coordenadas válidas.")
            _render_map(_map_df, cat_cols_chart)
        else:
            st.warning("Colunas lat/lon encontradas, mas sem valores válidos.")

    # ── Estratégia 2: coluna CEP → BrasilAPI (gratuito, sem chave) ──
    elif _cep_c:
        st.info(f"📮 Coluna de CEP detectada: **{_cep_c}**")
        _raw_ceps = df_view[_cep_c].dropna().astype(str)
        _clean_ceps_set = set(re.sub(r"[^0-9]", "", c)[:8] for c in _raw_ceps)
        _valid_ceps = tuple(sorted(c for c in _clean_ceps_set if len(c) == 8))

        if not _valid_ceps:
            st.error("Nenhum CEP com 8 dígitos encontrado. Verifique a formatação (ex: 01310-100 ou 01310100).")
        else:
            _n_ceps = len(_valid_ceps)
            st.markdown(
                f"**{_n_ceps} CEPs únicos** encontrados. "
                "Geocodificação via **BrasilAPI** (gratuita, sem necessidade de chave)."
            )
            _ceps_to_geo = _valid_ceps[:300]
            if _n_ceps > 300:
                st.warning(f"Muitos CEPs únicos ({_n_ceps}). Geocodificando os primeiros 300.")

            if st.button("📍 Geocodificar e Plotar no Mapa", key="btn_geo_cep", width='stretch'):
                with st.spinner(f"Geocodificando {len(_ceps_to_geo)} CEPs…"):
                    _geo_result = geocode_ceps(_ceps_to_geo)
                st.session_state["_geo_cep_cache"] = _geo_result
                st.rerun()

            if st.session_state.get("_geo_cep_cache"):
                _geo = st.session_state["_geo_cep_cache"]
                _cep_norm = df_view[_cep_c].astype(str).apply(
                    lambda x: re.sub(r"[^0-9]", "", x)[:8]
                )
                _map_rows = []
                for _idx in df_view.index:
                    _k = _cep_norm.loc[_idx]
                    _g = _geo.get(_k)
                    if not _g:
                        continue
                    _row = {"lat": _g["lat"], "lon": _g["lon"]}
                    _row["rua"] = (
                        str(df_view.at[_idx, _rua_c]) if _rua_c and _rua_c in df_view.columns
                        else _g.get("rua", "")
                    )
                    _row["bairro"] = _g.get("bairro", "")
                    _row["cidade"] = _g.get("cidade", "")
                    _row["estado"] = _g.get("estado", "")
                    if _nome_c and _nome_c in df_view.columns:
                        _row["nome"] = str(df_view.at[_idx, _nome_c])
                    for _cc in cat_cols_chart:
                        if _cc in df_view.columns:
                            _row[_cc] = df_view.at[_idx, _cc]
                    _map_rows.append(_row)

                if _map_rows:
                    _map_df = pd.DataFrame(_map_rows)
                    st.success(f"✅ {len(_map_df):,} registros plotados ({len(_geo)} CEPs resolvidos de {_n_ceps} únicos).")
                    _render_map(_map_df, cat_cols_chart)
                else:
                    st.error("Nenhum registro pôde ser geocodificado. BrasilAPI pode não ter coordenadas para esses CEPs.")

    # ── Estratégia 3: seleção manual de colunas ──
    else:
        st.warning("Nenhuma coluna de lat/lon ou CEP detectada automaticamente.")
        st.markdown("**Configure as colunas de endereço para plotar o mapa:**")
        _mc1, _mc2 = st.columns(2)
        _all_map_cols = ["(nenhuma)"] + df_view.columns.tolist()
        _sel_lat = _mc1.selectbox("Coluna Latitude", _all_map_cols, key="man_lat")
        _sel_lon = _mc2.selectbox("Coluna Longitude", _all_map_cols, key="man_lon")
        _sel_cep = _mc1.selectbox("Coluna CEP (alternativa)", _all_map_cols, key="man_cep")
        _sel_rua = _mc2.selectbox("Coluna Rua/Logradouro (tooltip)", _all_map_cols, key="man_rua")

        if _sel_lat != "(nenhuma)" and _sel_lon != "(nenhuma)":
            _map_df = pd.DataFrame()
            _map_df["lat"] = pd.to_numeric(df_view[_sel_lat], errors="coerce")
            _map_df["lon"] = pd.to_numeric(df_view[_sel_lon], errors="coerce")
            if _sel_rua != "(nenhuma)":
                _map_df["rua"] = df_view[_sel_rua].astype(str).values
            if _nome_c and _nome_c in df_view.columns:
                _map_df["nome"] = df_view[_nome_c].astype(str).values
            _map_df = _map_df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
            if not _map_df.empty:
                _render_map(_map_df, cat_cols_chart)
            else:
                st.error("Nenhum valor numérico válido nas colunas de lat/lon selecionadas.")

        elif _sel_cep != "(nenhuma)":
            _raw_m = df_view[_sel_cep].dropna().astype(str)
            _valid_m = tuple(sorted(set(
                c for c in (_raw_m.apply(lambda x: re.sub(r"[^0-9]", "", x)[:8]))
                if len(c) == 8
            )))[:300]
            if st.button("📍 Geocodificar por CEP", key="btn_geo_man", width='stretch'):
                with st.spinner(f"Geocodificando {len(_valid_m)} CEPs…"):
                    _geo_m = geocode_ceps(_valid_m)
                st.session_state["_geo_man_cache"] = {
                    "geo": _geo_m, "cep_col": _sel_cep, "rua_col": _sel_rua
                }
                st.rerun()

            if st.session_state.get("_geo_man_cache"):
                _mc_data = st.session_state["_geo_man_cache"]
                _geo_m = _mc_data["geo"]
                _cep_c2 = _mc_data["cep_col"]
                _rua_c2 = _mc_data["rua_col"]
                _cep_n2 = df_view[_cep_c2].astype(str).apply(lambda x: re.sub(r"[^0-9]", "", x)[:8])
                _map_rows2 = []
                for _idx in df_view.index:
                    _g = _geo_m.get(_cep_n2.loc[_idx])
                    if not _g:
                        continue
                    _row = {"lat": _g["lat"], "lon": _g["lon"],
                            "rua": (str(df_view.at[_idx, _rua_c2]) if _rua_c2 != "(nenhuma)" and _rua_c2 in df_view.columns
                                    else _g.get("rua", "")),
                            "bairro": _g.get("bairro", ""),
                            "cidade": _g.get("cidade", ""),
                            "estado": _g.get("estado", "")}
                    _map_rows2.append(_row)
                if _map_rows2:
                    _render_map(pd.DataFrame(_map_rows2), cat_cols_chart)
                else:
                    st.error("Nenhum CEP geocodificado. Verifique se os CEPs são válidos.")
        else:
            st.info("Selecione colunas de lat/lon ou CEP acima para visualizar o mapa.")

with tab_export:
    st.markdown("#### Exportar base consolidada")
    st.markdown(
        "Baixe os dados tratados para usar em outros sistemas ou para restaurar o hub depois.\n"
        "- **Parquet**: formato otimizado para reimportar neste hub (mais rápido).\n"
        "- **Excel**: apenas para bases pequenas (< 50 MB). Gera sob demanda."
    )

    exp1, exp2 = st.columns(2)
    exp2.download_button(
        "⬇️ Baixar Parquet",
        data=hub_para_bytes(df),
        file_name="hub_dados.parquet",
        mime="application/octet-stream",
    )
    if exp1.button("⬇️ Gerar e Baixar Excel"):
        if len(df) > 50_000:
            st.warning(f"Base grande ({len(df):,} linhas). Exportando primeiras 50.000 linhas para Excel.")
            df_xl = df.head(50_000)
        else:
            df_xl = df
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_xl.to_excel(writer, index=False, sheet_name="Hub")
        buf.seek(0)
        st.download_button(
            "Clique aqui para baixar",
            data=buf,
            file_name="hub_clientes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel_lazy",
        )

with tab_export:
    st.markdown("---")
    st.markdown("#### 🦆 Salvar no DuckDB (banco local)")
    if not _DUCKDB_OK:
        st.warning("DuckDB não instalado. Execute: `pip install duckdb`")
    else:
        db_path = st.text_input(
            "Arquivo do banco",
            value="dados.duckdb",
            help="Caminho do arquivo .duckdb onde os dados serão salvos",
            key="duckdb_path",
        )
        table_name = st.text_input(
            "Nome da tabela",
            value="dados",
            key="duckdb_table",
        )
        write_mode = st.radio(
            "Modo de escrita",
            ["Acumular (APPEND)", "Substituir (TRUNCATE)"],
            horizontal=True,
            key="duckdb_write_mode",
        )
        if st.button("💾 Salvar no DuckDB", key="btn_duckdb_save"):
            if not table_name.strip():
                st.error("Informe o nome da tabela.")
            else:
                try:
                    con = _duckdb.connect(db_path)
                    if write_mode.startswith("Substituir"):
                        con.execute(f"DROP TABLE IF EXISTS {table_name}")
                    con.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} AS
                        SELECT * FROM df WHERE 1=0
                    """)
                    t0 = time.time()
                    con.register("df", df)
                    con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                    con.unregister("df")
                    elapsed = time.time() - t0
                    rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    con.close()
                    st.success(
                        f"✅ {len(df):,} linhas salvas em `{db_path}` → tabela `{table_name}` "
                        f"(total na tabela: {rows:,} | {elapsed:.3f}s)"
                    )
                except Exception as _exc:
                    st.error(f"Erro ao salvar no DuckDB: {_exc}")

with st.expander("🤖 Assistente IA para Insights", expanded=False):
    st.markdown(
        "Gere uma leitura executiva dos seus dados usando inteligencia artificial.\n"
        "Sem chave de API, o assistente local analisa os dados e entrega insights automaticamente."
    )

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
# ANALISE ML
# ─────────────────────────────────────────────
st.divider()
st.markdown("## 🧠 Analise de Machine Learning")

if not rodar:
    tipo_label = tipo_problema.split("(")[0].strip()
    st.info(
        f"✅ Análise configurada: **{tipo_label}** | Alvo: **{target_col}** | "
        f"Preditores: **{len(features_selecionadas)}** variáveis.\n\n"
        "💡 **O modelo rodará automaticamente** quando você clicar em '▶ Atualizar Dashboard' na barra lateral."
    )
    st.stop()

# ─────────────────────────────────────────────
# Configurar MLflow para rastrear experimentos
# ─────────────────────────────────────────────
if _MLFLOW_OK:
    mlflow.set_tracking_uri("file:./mlruns")  # Local SQLite backend
    _ml_exp_name = f"ML_{tipo_problema.split('(')[0].strip()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(_ml_exp_name)
    mlflow.start_run(run_name=f"target={target_col}")
    mlflow.log_param("target_column", target_col)
    mlflow.log_param("n_features", len(features_selecionadas))
    mlflow.log_param("problem_type", tipo_problema)

if len(features_selecionadas) == 0:
    st.error(
        "**Nenhuma variavel numerica encontrada.**\n\n"
        "O modelo precisa de pelo menos uma coluna com numeros (ex: preco, idade, score). "
        "Verifique se sua planilha tem colunas numericas ou se a coluna alvo nao esta consumindo todas elas."
    )
    st.stop()

# Descarta apenas linhas sem valor na coluna alvo (não pode ser imputada)
_df_raw = df[features_selecionadas + [target_col]].dropna(subset=[target_col])

if _df_raw.empty or len(_df_raw) < 4:
    n_raw = len(df[features_selecionadas + [target_col]])
    st.error(
        f"**Dados insuficientes para treinar o modelo** (coluna alvo '{target_col}' tem menos de 4 valores preenchidos de {n_raw} linhas).\n\n"
        "Escolha outra coluna alvo que tenha mais dados preenchidos."
    )
    st.stop()

# Imputa NaN nas features com a mediana de cada coluna
_imputer = SimpleImputer(strategy="median")
_X_imputed = _imputer.fit_transform(_df_raw[features_selecionadas])
df_model = _df_raw.copy()
df_model[features_selecionadas] = _X_imputed

_n_imputed = int(_df_raw[features_selecionadas].isna().sum().sum())
if _n_imputed > 0:
    st.info(
        f"ℹ️ {_n_imputed:,} valores em branco nas colunas preditoras foram preenchidos automaticamente "
        "com a mediana de cada coluna para o treinamento do modelo."
    )

X = df_model[features_selecionadas]
le = LabelEncoder()
scaler = StandardScaler()

if tipo_problema == "Classificacao (sim/nao)":
    st.markdown(
        "**Classificacao** identifica se um cliente vai ou nao realizar uma acao (ex: contratar, comprar, churnar). "
        "O modelo aprende com o historico e estima a probabilidade para cada novo caso."
    )
    st.markdown("---")

    y_raw = df_model[target_col]
    y, _ = make_binary(y_raw)

    if len(np.unique(y)) < 2:
        st.error("A coluna alvo precisa ter ao menos 2 classes para classificacao. Escolha outra coluna alvo.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    modelos = {
        "Regressao Logistica": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    with st.spinner("Treinando modelos..."):
        resultados = []
        _sc_cls = StandardScaler()
        _X_tr_sc = _sc_cls.fit_transform(X_train)
        _X_te_sc = _sc_cls.transform(X_test)
        for nome, m in modelos.items():
            m.fit(X_train, y_train)
            acc = accuracy_score(y_test, m.predict(X_test))
            cv_folds = min(5, max(2, len(y) // 5))
            cv = cross_val_score(m, X, y, cv=cv_folds, scoring="accuracy").mean()
            resultados.append({"Modelo": nome, "Acuracia (teste)": acc, "Acuracia (cross-val)": cv})
            
            # Logar no MLflow
            if _MLFLOW_OK:
                mlflow.log_metric(f"{nome}_accuracy_test", acc)
                mlflow.log_metric(f"{nome}_accuracy_cv", cv)

        # Rede Neural (PyTorch) — lazy import
        try:
            import torch
            import torch.nn as nn
            _nn_cls = nn.Sequential(
                nn.Linear(len(features_selecionadas), 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            )
            _opt = torch.optim.Adam(_nn_cls.parameters(), lr=1e-3)
            _loss_fn = nn.BCELoss()
            _Xt = torch.tensor(_X_tr_sc, dtype=torch.float32)
            _yt = torch.tensor(y_train.astype(float), dtype=torch.float32).unsqueeze(1)
            _nn_cls.train()
            for _ in range(100):
                _opt.zero_grad()
                _loss_fn(_nn_cls(_Xt), _yt).backward()
                _opt.step()
            _nn_cls.eval()
            with torch.no_grad():
                _preds_nn = (_nn_cls(torch.tensor(_X_te_sc, dtype=torch.float32)).squeeze().numpy() >= 0.5).astype(int)
            _acc_nn = accuracy_score(y_test, _preds_nn)
            resultados.append({"Modelo": "Rede Neural (PyTorch)", "Acuracia (teste)": _acc_nn, "Acuracia (cross-val)": _acc_nn})
            
            # Logar no MLflow
            if _MLFLOW_OK:
                mlflow.log_metric("Rede Neural (PyTorch)_accuracy_test", _acc_nn)
                mlflow.log_metric("Rede Neural (PyTorch)_accuracy_cv", _acc_nn)
        except (ImportError, Exception):
            pass  # PyTorch não disponível

    df_res = pd.DataFrame(resultados).sort_values("Acuracia (teste)", ascending=False)
    melhor_nome = df_res.iloc[0]["Modelo"]
    
    # Logar melhor modelo no MLflow
    if _MLFLOW_OK:
        mlflow.log_metric("best_model_accuracy", df_res.iloc[0]["Acuracia (teste)"])
        mlflow.log_param("best_model", melhor_nome)
    # Se a rede neural for a melhor, tenta usá-la; caso contrário, usa sklearn
    if melhor_nome == "Rede Neural (PyTorch)":
        try:
            import torch
            class _NNWrapper:
                def __init__(self, net, scaler):
                    self.net = net
                    self.sc = scaler
                def predict(self, X_in):
                    Xt = torch.tensor(self.sc.transform(X_in), dtype=torch.float32)
                    with torch.no_grad():
                        return (self.net(Xt).squeeze().numpy() >= 0.5).astype(int)
                def predict_proba(self, X_in):
                    Xt = torch.tensor(self.sc.transform(X_in), dtype=torch.float32)
                    with torch.no_grad():
                        p = self.net(Xt).squeeze().numpy()
                    return np.column_stack([1 - p, p])
            melhor_modelo = _NNWrapper(_nn_cls, _sc_cls)
        except Exception:
            melhor_modelo = modelos[modelos.keys().__iter__().__next__()]  # Fallback para primeiro modelo sklearn
    else:
        melhor_modelo = modelos[melhor_nome]

    st.markdown("#### Comparacao de modelos")
    st.caption("Tres algoritmos foram testados. O de maior acuracia e selecionado automaticamente.")
    st.dataframe(df_res.style.format({"Acuracia (teste)": "{:.1%}", "Acuracia (cross-val)": "{:.1%}"}), width='stretch')
    st.success(f"Melhor modelo: **{melhor_nome}** com **{df_res.iloc[0]['Acuracia (teste)']:.1%}** de acuracia")

    if hasattr(melhor_modelo, "feature_importances_"):
        imp = pd.DataFrame({
            "Variavel": features_selecionadas,
            "Importancia": melhor_modelo.feature_importances_
        }).sort_values("Importancia", ascending=False)

        st.markdown("#### Variáveis mais importantes")
        st.caption("Quanto maior a barra, mais aquela variável influencia a decisão do modelo.")
        
        _fig_imp = px.bar(
            imp.sort_values("Importancia"),
            x="Importancia",
            y="Variavel",
            orientation="h",
            title="Feature Importance - Classificação",
            color="Importancia",
            color_continuous_scale="Blues",
            hover_data={"Importancia": ":.4f"},
            height=350
        )
        _fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, hovermode="closest")
        st.plotly_chart(_fig_imp, width='stretch')
        del _fig_imp

    probs = melhor_modelo.predict_proba(X)[:, 1]
    df_rank = df_model.copy()
    df_rank["prob_contratar"] = probs
    df_rank["previsao"] = ["Sim" if p >= 0.5 else "Nao" for p in probs]
    df_rank = df_rank.sort_values("prob_contratar", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    st.markdown("#### Ranking de probabilidade")
    st.caption("Registros ordenados do mais ao menos propenso a contratar/converter.")
    st.dataframe(
        df_rank.style.format({"prob_contratar": "{:.1%}"}),
        width='stretch'
    )

    st.markdown("#### Simular novo registro")
    st.caption("Ajuste os valores abaixo para ver a previsao do modelo para um perfil hipotetico.")
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

elif tipo_problema == "Regressao (valor numerico)":
    st.markdown(
        "**Regressao** estima um valor numerico continuo (ex: preco, faturamento, score). "
        "O modelo aprende a partir do historico e projeta valores para cada registro."
    )
    st.markdown("---")

    y_num = pd.to_numeric(coerce_numeric_series(df_model[target_col]), errors="coerce")
    _valid_y_mask = y_num.notna()
    X_reg = X.loc[_valid_y_mask]
    y = y_num.loc[_valid_y_mask].astype(float).values

    if len(y) < 4:
        st.error(
            f"A coluna alvo '{target_col}' nao possui valores numericos suficientes para regressao "
            f"(validos: {int(_valid_y_mask.sum())}/{len(df_model)})."
        )
        st.stop()

    if len(X_reg) < 10:
        st.warning("Poucos dados para regressao. Ideal: pelo menos 30 registros para resultados confiaveis.")

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.25, random_state=42)

    modelos = {
        "Regressao Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    with st.spinner("Treinando modelos..."):
        resultados = []
        _sc_reg = StandardScaler()
        _y_mean = float(np.mean(y_train))
        _y_std = float(np.std(y_train)) or 1.0
        _X_tr_sc_r = _sc_reg.fit_transform(X_train)
        _X_te_sc_r = _sc_reg.transform(X_test)
        _y_tr_norm = (y_train - _y_mean) / _y_std
        for nome, m in modelos.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            resultados.append({"Modelo": nome, "MAE (erro medio)": mae, "R2 (ajuste)": r2})
            
            # Logar no MLflow
            if _MLFLOW_OK:
                mlflow.log_metric(f"{nome}_mae", mae)
                mlflow.log_metric(f"{nome}_r2", r2)

        # Rede Neural (PyTorch) — lazy import
        try:
            import torch
            import torch.nn as nn
            _nn_reg = nn.Sequential(
                nn.Linear(len(features_selecionadas), 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )
            _opt_r = torch.optim.Adam(_nn_reg.parameters(), lr=1e-3)
            _loss_fn_r = nn.MSELoss()
            _Xt_r = torch.tensor(_X_tr_sc_r, dtype=torch.float32)
            _yt_r = torch.tensor(_y_tr_norm, dtype=torch.float32).unsqueeze(1)
            _nn_reg.train()
            for _ in range(150):
                _opt_r.zero_grad()
                _loss_fn_r(_nn_reg(_Xt_r), _yt_r).backward()
                _opt_r.step()
            _nn_reg.eval()
            with torch.no_grad():
                _pred_nn_r = _nn_reg(torch.tensor(_X_te_sc_r, dtype=torch.float32)).squeeze().numpy() * _y_std + _y_mean
            _mae_nn = mean_absolute_error(y_test, _pred_nn_r)
            _r2_nn = r2_score(y_test, _pred_nn_r)
            resultados.append({"Modelo": "Rede Neural (PyTorch)", "MAE (erro medio)": _mae_nn, "R2 (ajuste)": _r2_nn})
            
            # Logar no MLflow
            if _MLFLOW_OK:
                mlflow.log_metric("Rede Neural (PyTorch)_mae", _mae_nn)
                mlflow.log_metric("Rede Neural (PyTorch)_r2", _r2_nn)
        except (ImportError, Exception):
            pass  # PyTorch não disponível

    df_res = pd.DataFrame(resultados).sort_values("R2 (ajuste)", ascending=False)
    melhor_nome = df_res.iloc[0]["Modelo"]
    
    # Logar melhor modelo no MLflow
    if _MLFLOW_OK:
        mlflow.log_metric("best_model_r2", df_res.iloc[0]["R2 (ajuste)"])
        mlflow.log_metric("best_model_mae", df_res.iloc[0]["MAE (erro medio)"])
        mlflow.log_param("best_model", melhor_nome)
    if melhor_nome == "Rede Neural (PyTorch)":
        try:
            import torch
            class _NNRegWrapper:
                def __init__(self, net, scaler, y_mean, y_std):
                    self.net = net
                    self.sc = scaler
                    self.ym = y_mean
                    self.ys = y_std
                def predict(self, X_in):
                    Xt = torch.tensor(self.sc.transform(X_in), dtype=torch.float32)
                    with torch.no_grad():
                        return self.net(Xt).squeeze().numpy() * self.ys + self.ym
            melhor_modelo = _NNRegWrapper(_nn_reg, _sc_reg, _y_mean, _y_std)
        except Exception:
            melhor_modelo = modelos[modelos.keys().__iter__().__next__()]  # Fallback para primeiro modelo sklearn
    else:
        melhor_modelo = modelos[melhor_nome]

    st.markdown("#### Comparacao de modelos")
    st.caption("MAE = erro medio absoluto (menor e melhor). R2 = qualidade do ajuste (mais perto de 1 e melhor).")
    st.dataframe(df_res.style.format({"MAE (erro medio)": "{:.2f}", "R2 (ajuste)": "{:.2f}"}), width='stretch')
    st.success(f"Melhor modelo: **{melhor_nome}** com R2 = **{df_res.iloc[0]['R2 (ajuste)']:.2f}**")

    if hasattr(melhor_modelo, "feature_importances_"):
        imp = pd.DataFrame({
            "Variavel": features_selecionadas,
            "Importancia": melhor_modelo.feature_importances_
        }).sort_values("Importancia", ascending=False)

        st.markdown("#### Variáveis mais importantes")
        st.caption("Quanto maior a barra, mais aquela variável influencia o valor previsto.")
        
        _fig_imp_reg = px.bar(
            imp.sort_values("Importancia"),
            x="Importancia",
            y="Variavel",
            orientation="h",
            title="Feature Importance - Regressão",
            color="Importancia",
            color_continuous_scale="Oranges",
            hover_data={"Importancia": ":.4f"},
            height=350
        )
        _fig_imp_reg.update_layout(yaxis={"categoryorder": "total ascending"}, hovermode="closest")
        st.plotly_chart(_fig_imp_reg, width='stretch')
        del _fig_imp_reg

    pred_test = melhor_modelo.predict(X_test)
    st.markdown("#### Real vs Previsto")
    st.caption("Pontos próximos da linha vermelha indicam previsões precisas. Use zoom e pan para explorar.")
    
    _df_pred = pd.DataFrame({
        "Real": y_test,
        "Previsto": pred_test,
        "Erro": np.abs(y_test - pred_test)
    })
    
    _fig_real_pred = px.scatter(
        _df_pred,
        x="Real",
        y="Previsto",
        color="Erro",
        color_continuous_scale="Reds",
        hover_data={"Real": ":.2f", "Previsto": ":.2f", "Erro": ":.2f"},
        title="Real vs Previsto (com erro absoluto)",
        height=550,
        labels={"Real": "Valor Real", "Previsto": "Valor Previsto"}
    )
    
    _fig_real_pred.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode="lines",
        name="Perfeito (y=x)",
        line=dict(color="red", dash="dash", width=2),
        hovertemplate="Linha perfeita<extra></extra>"
    ))
    
    _fig_real_pred.update_layout(hovermode="closest")
    st.plotly_chart(_fig_real_pred, width='stretch')
    del _fig_real_pred, _df_pred

    previsoes_todas = melhor_modelo.predict(X)
    df_rank = df_model.copy()
    df_rank[f"previsao_{target_col}"] = previsoes_todas
    df_rank = df_rank.sort_values(f"previsao_{target_col}", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    st.markdown(f"#### Ranking por {target_col} previsto")
    st.dataframe(df_rank, width='stretch')

elif tipo_problema == "Clustering (agrupamento)":
    st.markdown(
        "**Clustering** descobre grupos naturais nos dados sem precisar de uma coluna alvo. "
        "Ideal para segmentar clientes, identificar perfis e criar estrategias diferenciadas por grupo."
    )
    st.markdown("---")

    X_scaled = scaler.fit_transform(X)

    st.markdown("#### Configuracao do agrupamento")
    n_clusters = st.slider(
        "Numero de grupos (clusters)",
        min_value=2, max_value=8, value=3,
        help="Comece com 3 grupos. Aumente se os perfis parecerem muito misturados."
    )

    with st.spinner("Agrupando registros..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Logar no MLflow
        if _MLFLOW_OK:
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_metric("kmeans_inertia", kmeans.inertia_)

    df_cluster = df_model.copy()
    df_cluster["Grupo"] = [f"Grupo {l+1}" for l in labels]

    st.markdown("#### Perfil medio por grupo")
    st.caption("Compare os grupos para entender o que diferencia cada segmento.")
    perfil = df_cluster.groupby("Grupo")[features_selecionadas].mean().round(2)
    st.dataframe(perfil, width='stretch')

    st.markdown("#### Tamanho dos grupos")
    tamanhos = df_cluster["Grupo"].value_counts().reset_index()
    tamanhos.columns = ["Grupo", "Qtd. Clientes"]
    
    _fig_tamanhos = px.bar(
        tamanhos,
        x="Grupo",
        y="Qtd. Clientes",
        color="Qtd. Clientes",
        color_continuous_scale="Viridis",
        title="Distribuição de Clientes por Grupo",
        hover_data={"Qtd. Clientes": ":,"},
        height=400
    )
    _fig_tamanhos.update_layout(hovermode="closest")
    st.plotly_chart(_fig_tamanhos, width='stretch')
    del _fig_tamanhos

    if len(features_selecionadas) >= 2:
        st.markdown("#### Visualização dos grupos (Interativa 3D)")
        st.caption("Selecione variáveis para ver como os grupos se distribuem. Escolha Z para ativar modo 3D.")
        col_x, col_y, col_z = st.columns(3)
        fx = col_x.selectbox("Eixo X", features_selecionadas, index=0)
        fy = col_y.selectbox("Eixo Y", features_selecionadas, index=min(1, len(features_selecionadas)-1))
        fz = col_z.selectbox("Eixo Z (3D)", ["Nenhum"] + features_selecionadas, index=0)
        
        _cluster_plot_df = df_cluster.dropna(subset=[fx, fy] + ([fz] if fz != "Nenhum" else []))
        
        if fz != "Nenhum" and len(_cluster_plot_df) > 0:
            _fig_cluster = px.scatter_3d(
                _cluster_plot_df,
                x=fx, y=fy, z=fz,
                color="Grupo",
                hover_data={fx: ":.2f", fy: ":.2f", fz: ":.2f", "Grupo": True},
                title="Distribuição dos Clusters em 3D",
                height=700
            )
        else:
            _fig_cluster = px.scatter(
                _cluster_plot_df,
                x=fx, y=fy,
                color="Grupo",
                hover_data={fx: ":.2f", fy: ":.2f", "Grupo": True},
                title="Distribuição dos Clusters",
                height=600
            )
        
        _fig_cluster.update_layout(hovermode="closest")
        st.plotly_chart(_fig_cluster, width='stretch')
        del _fig_cluster, _cluster_plot_df

    st.markdown("#### Dados com grupos atribuidos")
    st.dataframe(df_cluster, width='stretch')
    st.info("Dica: exporte os dados acima (aba Exportacao) e use a coluna Grupo para criar estrategias diferenciadas para cada segmento.")

# ─────────────────────────────────────────────
# Encerrar MLflow e adicionar link para visualização
# ─────────────────────────────────────────────
if _MLFLOW_OK:
    mlflow.end_run()
    
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(
            "📊 Ver todos os experimentos no MLflow",
            width='stretch',
            help="Abre a interface MLflow em localhost:5000 para visualizar histórico de experimentos, comparar modelos e rastrear métricas."
        ):
            st.info(
                "Para visualizar os experimentos no MLflow, execute no terminal:\n\n"
                "`mlflow ui --backend-store-uri file:./mlruns`\n\n"
                "Depois acesse: http://localhost:5000"
            )
    
    st.caption(
        f"✅ Experimento rastreado automaticamente. "
        f"Os dados estão salvos em `./mlruns/` para reprodução e comparação futura."
    )

