# Zaawansowany Pipeline Produkcyjny

Ten dokument opisuje dodatkowe kroki związane z integracją produkcyjną, wdrożeniem modelu ML i utrzymaniem systemu na żywo.

## 1. Integracja z Realnymi Danymi

- **Źródła Danych:** 
  - Bazy danych transakcyjnych, logi serwera, systemy monitorujące.
  - API firmowych systemów finansowych.
- **Proces ETL (Extract, Transform, Load):**
  - Ekstrakcja danych z różnych źródeł.
  - Transformacja danych przy użyciu narzędzi np. Apache Airflow lub Apache NiFi.
  - Ładowanie przetworzonych danych do hurtowni danych (np. BigQuery, Redshift) lub do bezpośredniego wykorzystania przez model.
- **Przykładowy DAG w Apache Airflow:**
```python
# from airflow import DAG
# from airflow.operators.bash import BashOperator
# from datetime import datetime
# 
# default_args = {
#     'owner': 'olafcio42',
#     'start_date': datetime(2025, 5, 29),
#     'retries': 1,
# }
# 
# with DAG(dag_id='etl_pipeline', default_args=default_args, schedule_interval='@hourly') as dag:
#     extract = BashOperator(
#         task_id='extract_data',
#         bash_command='python extract.py'
#     )
#     transform = BashOperator(
#         task_id='transform_data',
#         bash_command='python transform.py'
#     )
#     load = BashOperator(
#         task_id='load_data',
#         bash_command='python load.py'
#     )
#     
#     extract >> transform >> load
```
- **Monitorowanie Jakości Danych:** 
  - Integracja z narzędziami takimi jak Great Expectations pozwala na weryfikację jakości danych przed ich załadowaniem.

## 2. Wdrożenie Modelu w Środowisku Produkcyjnym

- **Konteneryzacja i Docker:**
  - Model oraz API są pakowane w obrazy Docker, co ułatwia przenoszenie między środowiskami.
- **Przykładowy Dockerfile:**
```dockerfile
# Przykładowy Dockerfile dla ml_risk_api
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["uvicorn", "final_risk_api:app", "--host", "0.0.0.0", "--port", "5000"]
```
- **Orkiestracja z Kubernetes:**
  - Wdrożenie aplikacji w klastrze Kubernetes dla zapewnienia wysokiej dostępności i skalowalności.
- **Przykładowy plik deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-risk-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-risk-api
  template:
    metadata:
      labels:
        app: ml-risk-api
    spec:
      containers:
      - name: ml-risk-api
        image: your_docker_registry/ml_risk_api:latest
        ports:
        - containerPort: 5000
```
- **Skalowanie Automatyczne:**
  - Użycie Horizontal Pod Autoscaler (HPA) w Kubernetes pomaga w automatycznym skalowaniu aplikacji w zależności od obciążenia.

## 3. Monitorowanie i Utrzymanie Pipeline’u

- **Logowanie i Monitorowanie:**
  - Zbiór logów z wykorzystaniem ELK Stack (Elasticsearch, Logstash, Kibana) lub alternatywnie Prometheus & Grafana.
  - Monitorowanie przepływu danych i jakości prognoz modelu.
- **Automatyczny Retraining:**
  - Wdrożenie mechanizmu retrainingu, który wywoływany jest automatycznie, gdy następuje drift danych lub spadek wydajności modelu.
  - Testowanie modelu w środowisku staging przed przeprowadzeniem _rolloutu_ do produkcji.

## 4. CI/CD dla Pipeline’u Produkcyjnego

- **Automatyzacja Testowania, Budowania i Wdrażania:**
  - Użycie narzędzi CI/CD (np. GitHub Actions, Jenkins, GitLab CI) do:
    - Testowania commitów i integracji nowych funkcjonalności.
    - Budowania obrazów Docker.
    - Wdrażania aplikacji na środowisko staging lub produkcyjne.
- **Przykładowy workflow GitHub Actions:**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests
      - name: Build Docker image
        run: |
          docker build -t your_docker_registry/ml_risk_api:latest .
      - name: Push Docker image
        run: |
          echo $DOCKER_PASSWORD | docker login --username $DOCKER_USERNAME --password-stdin
          docker push your_docker_registry/ml_risk_api:latest
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f deployment.yaml
```