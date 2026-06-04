# This file extracts raw data from MySQL database or reads a CSV file if provided

import pandas as pd  # data manipulation library
from sqlalchemy import create_engine  # database connection engine
from dotenv import load_dotenv  # load environment variables
from pathlib import Path
import os  # access environment variables

load_dotenv()  # load .env file variables into environment


def resolve_path(path: str):
    if not path:
        return None
    file_path = Path(path.strip())
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent.parent / file_path
    return file_path


def get_engine():
    # build mysql connection string using env variables
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)  # return sqlalchemy engine object


def extract_data(csv_path: str = None):
    csv_path = csv_path or os.getenv('DATA_CSV_PATH')
    if csv_path:
        csv_file = resolve_path(csv_path)
        if csv_file and csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"Extracted {len(df)} rows from CSV {csv_file}")
            return df
        raise FileNotFoundError(
            f"CSV path '{csv_file}' not found. Set DATA_CSV_PATH in .env or pass a valid file path."
        )

    engine = get_engine()  # get database connection
    df = pd.read_sql("SELECT * FROM online_shoppers", engine)  # fetch all rows from table
    print(f"Extracted {len(df)} rows")  # log how many rows fetched
    return df  # return dataframe


if __name__ == "__main__":
    df = extract_data()  # run extraction
    print(df.head())  # print first 5 rows to verify
