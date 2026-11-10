import pandas as pd
import sqlite3
import os
import zipfile
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from kaggle.api.kaggle_api_extended import KaggleApi

# Default arguments for the DAG
default_args = {
    'owner': 'HenHei',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 6),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'world_happiness_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for World Happiness Report data with validation and trigger rules',
    schedule_interval='@daily',
)

# File paths
csv_file_path = '/home/de2023a/airflow/datasets/2019.csv'
db_path = '/home/de2023a/airflow/databases/happiness_data.db'

# Task 1: Extract data
def extract_data(**kwargs):
    # Set up Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset file
    api.dataset_download_file('unsdsn/world-happiness', file_name='2019.csv', path='/home/de2023a/airflow/datasets')

    # Define file paths
    downloaded_file_path = '/home/de2023a/airflow/datasets/2019.csv'
    zip_file_path = downloaded_file_path + '.zip'

    # Check if the downloaded file is a ZIP file
    if zipfile.is_zipfile(zip_file_path):
        # If it's a ZIP file, unzip it
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('/home/de2023a/airflow/datasets')
        # Update file path to extracted CSV
        os.remove(zip_file_path)  # Optionally delete the ZIP file after extraction
    else:
        print("Downloaded file is not a ZIP archive, skipping extraction.")

    # Push the CSV file path to XCom for use in the next steps
    kwargs['ti'].xcom_push(key='csv_file_path', value=downloaded_file_path)

extract_task = PythonOperator(
    task_id='extract_task',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

# Task 2: Transform data
def transform_data(**kwargs):
    # Retrieve file path from XCom
    file_path = kwargs['ti'].xcom_pull(key='csv_file_path')
    df = pd.read_csv(file_path)

    # Handle missing values in critical columns
    critical_columns = ['GDP per capita', 'Social support']
    for column in critical_columns:
        df[column].fillna(df[column].median(), inplace=True)

    # Calculate global average happiness score
    global_average_happiness = df['Score'].mean()

    # Calculate "happiness_gap" as the difference between each country's score and the global average
    df['happiness_gap'] = df['Score'] - global_average_happiness

    # Save the transformed data to a new CSV file and pass file path to XCom
    transformed_file_path = '/tmp/transformed_world_happiness_data.csv'
    df.to_csv(transformed_file_path, index=False)
    kwargs['ti'].xcom_push(key='transformed_file_path', value=transformed_file_path)

transform_task = PythonOperator(
    task_id='transform_task',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

# Task 3: Validate data
def validate_data(**kwargs):
    # Retrieve transformed file path from XCom
    transformed_file_path = kwargs['ti'].xcom_pull(key='transformed_file_path')
    df = pd.read_csv(transformed_file_path)

    # Ensure no critical columns have missing values after transformation
    critical_columns = ['GDP per capita', 'Social support', 'happiness_gap']
    if df[critical_columns].isnull().any().any():
        raise ValueError("Validation failed: Missing values in critical columns after transformation.")

    # Pass the file path to XCom for the next task
    kwargs['ti'].xcom_push(key='validated_file_path', value=transformed_file_path)

validate_task = PythonOperator(
    task_id='validate_task',
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)

# Task 4: Load data
def load_data(**kwargs):
    # Retrieve validated file path from XCom
    validated_file_path = kwargs['ti'].xcom_pull(key='validated_file_path')
    df = pd.read_csv(validated_file_path)

    # Load data into SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor() # A cursor is an object used to interact with a database. It acts as a control structure that enables traversal over the records in a database and execution of SQL commands. The cursor is part of the database connection and is used to execute SQL queries, fetch data, and manage database operations.

    # Create table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS world_happiness_2019 (
            "Overall rank" INTEGER,
            "Country or region" VARCHAR(100),
            Score FLOAT,
            "GDP per capita" FLOAT,
            "Social Support" FLOAT,
            "Healthy life expectancy" FLOAT,
            "Freedom to make life choices" FLOAT,
            Generosity FLOAT,
            "Perceptions of corruption" FLOAT,
            happiness_gap FLOAT
        )
    ''')

    # Insert data into the database
    df.to_sql('world_happiness_2019', conn, if_exists='append', index=False)
    conn.commit()
    conn.cursor()
    conn.close()

load_task = PythonOperator(
    task_id='load_task',
    python_callable=load_data,
    provide_context=True,
    trigger_rule='all_success',  # Ensures load_task only runs if validate_task is successful
    dag=dag,
)

# Set task dependencies with trigger rules
extract_task >> transform_task >> validate_task >> load_task