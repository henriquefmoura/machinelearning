import argparse
import os
import time

import duckdb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Carrega CSV em DuckDB e executa query agregada."
    )
    parser.add_argument("--db", default=os.getenv("DUCKDB_PATH", "dados.duckdb"),
                        help="Caminho do arquivo .duckdb (default: dados.duckdb)")
    parser.add_argument("--table", default=os.getenv("BQ_TABLE", ""),
                        help="Nome da tabela destino")
    parser.add_argument("--file-path", default=os.getenv("BQ_FILE_PATH", "dados.csv"),
                        help="Caminho do arquivo CSV")
    parser.add_argument("--delimiter", default=os.getenv("BQ_DELIMITER", ","),
                        help="Separador do CSV (default: ,)")
    parser.add_argument("--write-disposition", default=os.getenv("BQ_WRITE_DISPOSITION", "WRITE_APPEND"),
                        choices=["WRITE_APPEND", "WRITE_TRUNCATE"],
                        help="WRITE_APPEND: acumula | WRITE_TRUNCATE: substitui (default: WRITE_APPEND)")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.table:
        raise ValueError(
            "Parâmetro obrigatório ausente: --table. "
            "Preencha via argumento ou variável BQ_TABLE."
        )
    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {args.file_path}")


def upload_csv(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    print(f"Carregando {args.file_path} na tabela '{args.table}'...")

    if args.write_disposition == "WRITE_TRUNCATE":
        con.execute(f"DROP TABLE IF EXISTS {args.table}")

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {args.table} AS
        SELECT * FROM read_csv_auto('{args.file_path}', delim='{args.delimiter}')
        WHERE 1=0
    """)
    con.execute(f"""
        INSERT INTO {args.table}
        SELECT * FROM read_csv_auto('{args.file_path}', delim='{args.delimiter}')
    """)
    print("Carregamento concluído!")


def run_query(con: duckdb.DuckDBPyConnection, table: str) -> None:
    query = f"""
    SELECT
        categoria,
        COUNT(*) AS total,
        AVG(valor) AS media
    FROM {table}
    GROUP BY categoria
    ORDER BY total DESC
    """

    print("Executando query...")
    start = time.time()
    df = con.execute(query).df()
    elapsed = time.time() - start

    print(df.head())
    print(f"Tempo de execução: {elapsed:.4f} segundos")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        con = duckdb.connect(args.db)
        upload_csv(con, args)
        run_query(con, args.table)
        con.close()
        return 0
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
