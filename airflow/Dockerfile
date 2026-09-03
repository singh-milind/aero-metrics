FROM apache/airflow:3.1.0-python3.13

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    dvc \
    numpy==2.5.2 \
    pandas==3.0.5 \
    scikit-learn==1.9.0 \
    scipy==1.18.1 \
    xgboost==3.4.1 \
    shap==0.52.0 \
    joblib==1.5.3 \
    requests==2.34.2 \
    psycopg2-binary==2.9.12 \
    sqlalchemy==2.0.52 \
    python-dotenv==1.2.3