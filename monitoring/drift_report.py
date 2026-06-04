# This file generates data drift report using Evidently AI

import pandas as pd  # data manipulation
from evidently.report import Report  # evidently report class
from evidently.metric_preset import DataDriftPreset  # preset for data drift metrics
from sqlalchemy import create_engine  # database connection
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

def get_engine():
    # build mysql connection string
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)  # return engine

def generate_drift_report():
    engine = get_engine()  # get db connection

    # load full transformed data
    df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)

    # use first 70% as reference (training data)
    reference = df.iloc[:int(len(df) * 0.7)]

    # use last 30% as current (new incoming data)
    current = df.iloc[int(len(df) * 0.7):]

    print(f"Reference size: {len(reference)} rows")  # log reference size
    print(f"Current size: {len(current)} rows")      # log current size

    # create evidently report with data drift preset
    report = Report(metrics=[DataDriftPreset()])

    # run report on reference vs current data
    report.run(reference_data=reference, current_data=current)

    # save report as html file
    os.makedirs('./monitoring', exist_ok=True)  # create monitoring dir if not exists
    report.save_html('./monitoring/drift_report.html')  # save html report
    print("Drift report saved to ./monitoring/drift_report.html")  # log save

if __name__ == "__main__":
    generate_drift_report()  # run drift report generation