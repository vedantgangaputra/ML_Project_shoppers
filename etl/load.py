# This file loads transformed data into PostgreSQL data warehouse

import pandas as pd  # data manipulation
from sqlalchemy import create_engine  # database connection engine
from dotenv import load_dotenv  # load environment variables
import os  # access environment variables

load_dotenv()  # load .env file

def get_mysql_engine():
    # build mysql connection string for source database
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)  # return mysql engine

def get_pg_engine():
    # build postgresql connection string for data warehouse
    url = (
        f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_NAME')}"
    )
    return create_engine(url)  # return postgresql engine

def load_data(df):
    # load transformed data into postgresql warehouse
    pg_engine = get_pg_engine()  # get postgresql connection
    df.to_sql(
        'transformed_shoppers',  # table name in postgresql
        pg_engine,               # postgresql engine
        if_exists='replace',     # drop and recreate table every time
        index=False              # do not write index column
    )
    print(f"Loaded {len(df)} rows into PostgreSQL warehouse")  # log success

if __name__ == "__main__":
    from extract import extract_data      # import extract function
    from transform import transform_data  # import transform function
    df = extract_data()                   # extract raw data from mysql
    df = transform_data(df)              # clean and encode data
    load_data(df)                        # load into postgresql warehouse