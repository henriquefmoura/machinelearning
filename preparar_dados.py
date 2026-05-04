# /// script
# requires-python = ">=3.9"
# dependencies = ["pandas>=2.0", "openpyxl>=3.1", "pyarrow>=14.0"]
# ///
"""
preparar_dados.py — Converte planilha grande para Parquet comprimido.
Execute no seu próprio computador, não no app Streamlit.

USO:
    uv run preparar_dados.py ARQUIVO [--linhas N] [--saida DESTINO]

EXEMPLOS:
    uv run preparar_dados.py clientes.xlsx
    uv run preparar_dados.py vendas.csv --linhas 500000
    uv run preparar_dados.py base.xlsx --saida base_preparada.parquet

RESULTADO:
    Gera um arquivo .parquet comprimido (normalmente 5-10x menor que o CSV/Excel).
    Envie esse .parquet para o app ML Insights Hub no campo de upload.

REQUISITOS:
    - uv instalado (https://docs.astral.sh/uv/) OU Python com pandas/openpyxl/pyarrow
    - O script roda LOCALMENTE no seu PC, usando a memória da sua máquina.
"""

import sys
import argparse
import pathlib
import csv as _csv

import pandas as pd


def detectar_separador(caminho: pathlib.Path) -> str:
    with open(caminho, "rb") as fh:
        amostra = fh.read(16384).decode("utf-8", errors="ignore")
    try:
        dialect = _csv.Sniffer().sniff(amostra, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return ";" if amostra.count(";") > amostra.count(",") else ","


def ler_csv_chunks(caminho: pathlib.Path, max_linhas: int) -> pd.DataFrame:
    sep = detectar_separador(caminho)
    print(f"  Separador detectado: {repr(sep)}")
    partes = []
    total = 0
    for chunk in pd.read_csv(
        caminho,
        sep=sep,
        chunksize=100_000,
        low_memory=False,
        encoding="utf-8-sig",
        on_bad_lines="skip",
    ):
        partes.append(chunk)
        total += len(chunk)
        print(f"  {total:>10,} linhas lidas...", end="\r", flush=True)
        if max_linhas and total >= max_linhas * 2:
            print(f"\n  Limite de leitura atingido ({total:,} linhas). Amostrando...")
            break
    print()
    return pd.concat(partes, ignore_index=True)


def ler_excel(caminho: pathlib.Path) -> pd.DataFrame:
    print("  Carregando Excel — pode demorar para arquivos grandes...")
    return pd.read_excel(caminho)


def main():
    parser = argparse.ArgumentParser(
        description="Converte planilha grande para Parquet comprimido para envio ao ML Insights Hub"
    )
    parser.add_argument("arquivo", help="Caminho do arquivo .xlsx, .xls ou .csv")
    parser.add_argument(
        "--linhas",
        type=int,
        default=200_000,
        help="Máximo de linhas na amostra final (padrão: 200000). Use 0 para manter todas.",
    )
    parser.add_argument("--saida", default=None, help="Arquivo de saída (padrão: mesmo nome + .parquet)")
    args = parser.parse_args()

    entrada = pathlib.Path(args.arquivo)
    if not entrada.exists():
        print(f"ERRO: Arquivo não encontrado: {entrada}")
        sys.exit(1)

    saida = pathlib.Path(args.saida) if args.saida else entrada.with_suffix(".parquet")
    max_linhas = args.linhas

    print(f"\n{'='*60}")
    print(f"  Arquivo de entrada : {entrada}")
    print(f"  Arquivo de saída   : {saida}")
    print(f"  Limite de linhas   : {max_linhas:,}" if max_linhas else "  Limite de linhas   : sem limite")
    print(f"{'='*60}\n")

    ext = entrada.suffix.lower()
    if ext == ".csv":
        df = ler_csv_chunks(entrada, max_linhas)
    elif ext in (".xlsx", ".xls"):
        df = ler_excel(entrada)
    else:
        print(f"ERRO: Formato não suportado: {ext}. Use .csv, .xlsx ou .xls")
        sys.exit(1)

    total_lido = len(df)
    print(f"\nTotal lido: {total_lido:,} linhas × {len(df.columns)} colunas")

    if max_linhas and len(df) > max_linhas:
        print(f"Amostrando {max_linhas:,} linhas aleatórias (seed=42 para reproducibilidade)...")
        df = df.sample(max_linhas, random_state=42).reset_index(drop=True)

    print(f"Salvando como Parquet comprimido (snappy)...")
    df.to_parquet(saida, index=False, compression="snappy")

    tamanho_mb = saida.stat().st_size / (1024 * 1024)
    print(f"\n✅ Concluído!")
    print(f"   Linhas salvas  : {len(df):,} de {total_lido:,} lidas")
    print(f"   Arquivo gerado : {saida}")
    print(f"   Tamanho        : {tamanho_mb:.1f} MB")
    print(f"\nEnvie o arquivo '{saida.name}' no campo de upload do ML Insights Hub.")


if __name__ == "__main__":
    main()
