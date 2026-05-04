import argparse
import os
import time

from google.cloud import bigquery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faz upload de CSV para BigQuery e executa query agregada."
    )
    parser.add_argument("--credencial", default=os.getenv("BQ_CREDENCIAL", "credencial.json"))
    parser.add_argument("--project-id", default=os.getenv("BQ_PROJECT_ID", ""))
    parser.add_argument("--dataset", default=os.getenv("BQ_DATASET", ""))
    parser.add_argument("--table", default=os.getenv("BQ_TABLE", ""))
    parser.add_argument("--file-path", default=os.getenv("BQ_FILE_PATH", "dados.csv"))
    parser.add_argument("--location", default=os.getenv("BQ_LOCATION", "US"))
    parser.add_argument("--delimiter", default=os.getenv("BQ_DELIMITER", ","))
    parser.add_argument("--write-disposition", default=os.getenv("BQ_WRITE_DISPOSITION", "WRITE_APPEND"))
    parser.add_argument("--max-bad-records", type=int, default=int(os.getenv("BQ_MAX_BAD_RECORDS", "0")))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.project_id:
        missing.append("project-id")
    if not args.dataset:
        missing.append("dataset")
    if not args.table:
        missing.append("table")
    if missing:
        raise ValueError(
            "Parâmetros obrigatórios ausentes: " + ", ".join(missing) + ". "
            "Preencha via argumentos ou variáveis BQ_PROJECT_ID, BQ_DATASET, BQ_TABLE."
        )

    if not os.path.exists(args.credencial):
        raise FileNotFoundError(f"Arquivo de credencial não encontrado: {args.credencial}")

    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {args.file_path}")


def upload_csv(client: bigquery.Client, table_id: str, file_path: str, args: argparse.Namespace) -> None:
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        field_delimiter=args.delimiter,
        write_disposition=args.write_disposition,
        max_bad_records=args.max_bad_records,
    )

    print(f"Enviando arquivo {file_path} para {table_id}...")
    with open(file_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_id, location=args.location, job_config=job_config)

    job.result()
    print("Upload concluído!")


def run_query(client: bigquery.Client, table_id: str, location: str):
    query = f"""
    SELECT
        categoria,
        COUNT(*) AS total,
        AVG(valor) AS media
    FROM `{table_id}`
    GROUP BY categoria
    ORDER BY total DESC
    """

    print("Executando query...")
    start = time.time()
    df = client.query(query, location=location).to_dataframe()
    elapsed = time.time() - start

    print(df.head())
    print(f"Tempo de execução: {elapsed:.2f} segundos")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        table_id = f"{args.project_id}.{args.dataset}.{args.table}"
        client = bigquery.Client.from_service_account_json(args.credencial, project=args.project_id)

        upload_csv(client, table_id, args.file_path, args)
        run_query(client, table_id, args.location)
        return 0
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
