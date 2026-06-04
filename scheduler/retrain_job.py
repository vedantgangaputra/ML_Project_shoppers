# This file schedules automatic ETL + model retraining using APScheduler

from apscheduler.schedulers.blocking import BlockingScheduler  # blocking scheduler runs forever
from apscheduler.triggers.cron import CronTrigger  # cron trigger for scheduling
import sys  # system path manipulation
import os  # access env variables
from dotenv import load_dotenv  # load env variables

load_dotenv()  # load .env file

# add root and ml directory to path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ml')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../etl')))

from etl.extract import extract_data      # import extract function
from etl.transform import transform_data  # import transform function
from etl.load import load_data            # import load function
from ml.train import train                # import train function

def run_etl():
    # run full ETL pipeline — extract, transform, load
    print("ETL started...")       # log start
    df = extract_data()           # extract raw data from mysql
    df = transform_data(df)       # clean and encode data
    load_data(df)                 # load into transformed_shoppers table
    print("ETL completed")        # log completion

def run_training():
    # run ML training pipeline after ETL
    print("Training started...")  # log start
    train()                       # train model and log to mlflow
    print("Training completed")   # log completion

def run_pipeline():
    # run full pipeline — ETL then training
    print("Pipeline started...")  # log start
    run_etl()                     # run ETL first
    run_training()                # then retrain model
    print("Pipeline completed")   # log completion

if __name__ == "__main__":
    scheduler = BlockingScheduler()  # create scheduler instance

    # schedule pipeline to run every day at 2:00 AM
    scheduler.add_job(
        run_pipeline,              # function to run
        CronTrigger(hour=2, minute=0),  # every day at 2 AM
        id='retrain_job',          # unique job id
        name='ETL + Retrain Job',  # job name
        replace_existing=True      # replace if job already exists
    )

    print("Scheduler started — pipeline will run daily at 2:00 AM")  # log start
    
    try:
        scheduler.start()          # start scheduler — runs forever
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped") # log stop on ctrl+c