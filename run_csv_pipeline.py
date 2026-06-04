from dotenv import load_dotenv
from pathlib import Path
import os
import argparse

load_dotenv()


def resolve_path(path: str) -> Path:
    file_path = Path(path.strip())
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent / file_path
    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Train the project from a raw CSV file and prepare MLflow/UI instructions."
    )
    parser.add_argument(
        "--csv",
        help="Path to the online shoppers CSV file",
        default=os.getenv("DATA_CSV_PATH")
    )
    parser.add_argument(
        "--mlflow-uri",
        help="MLflow tracking URI or local folder",
        default=os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    )
    parser.add_argument(
        "--model-path",
        help="Path to save the trained model",
        default=os.getenv("MODEL_PATH", "./ml/model.pkl")
    )
    args = parser.parse_args()

    if not args.csv:
        raise SystemExit(
            "CSV file path is required. Set DATA_CSV_PATH in .env or pass --csv."
        )

    csv_file = resolve_path(args.csv)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    os.environ["DATA_CSV_PATH"] = str(csv_file)
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    os.environ["MODEL_PATH"] = args.model_path

    print(f"Using CSV: {csv_file}")
    print(f"MLflow tracking URI: {args.mlflow_uri}")
    print(f"Model path: {args.model_path}")
    print("Starting training...\n")

    from ml.train import train
    train()

    mlflow_uri = args.mlflow_uri
    if not mlflow_uri.startswith("http") and not mlflow_uri.startswith("file://"):
        mlflow_uri = f"file:///{Path(mlflow_uri).resolve().as_posix()}"

    print("\nTraining complete.")
    print("Run the UI servers next:")
    print(f"  mlflow ui --backend-store-uri \"{mlflow_uri}\" --port 5000")
    print("  uvicorn api.static.main:app --reload --port 8000")
    print("\nOpen in your browser:")
    print("  http://127.0.0.1:8000  (ML app frontend)")
    print("  http://127.0.0.1:5000  (MLflow dashboard)")


if __name__ == "__main__":
    main()
