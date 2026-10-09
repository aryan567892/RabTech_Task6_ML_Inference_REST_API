# RabTech Task 6 — Real-Time ML Inference REST API & Capstone

A production-style machine-learning inference microservice built with **FastAPI**, **scikit-learn**, **Docker**, and **pytest**.

## What this project demonstrates

- A trained and serialized **champion ML model**
- REST inference through a `/predict` endpoint
- JSON request validation with Pydantic
- Prediction probabilities in the response
- Health checking through `/health`
- Docker containerization with pinned dependencies
- Automated API unit tests
- End-to-end architecture documentation

## Model

The repository uses the scikit-learn Iris dataset as a compact, reproducible demonstration model.
Two classifiers were compared on a stratified holdout set:

- Logistic Regression
- Random Forest

**Selected champion:** `logistic_regression`

Validation scores are stored in `model/metadata.json`. The final champion is then retrained on the complete Iris dataset before being packaged for inference.

## Project structure

```text
RabTech_Task6_ML_Inference_REST_API/
├── app/
│   ├── __init__.py
│   └── main.py
├── model/
│   ├── champion_model.joblib
│   └── metadata.json
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Run locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{"sepal_length_cm":5.1,"sepal_width_cm":3.5,"petal_length_cm":1.4,"petal_width_cm":0.2}"
```

Example JSON response:

```json
{
  "predicted_class": "setosa",
  "predicted_class_index": 0,
  "probabilities": [0.99, 0.01, 0.0]
}
```

## Run tests

```bash
pytest -q
```

Tests cover:

1. Health endpoint
2. Valid prediction schema
3. Prediction probability output
4. Invalid input validation and HTTP 422 response

## Docker

Build:

```bash
docker build -t rabtech-ml-api .
```

Run:

```bash
docker run --rm -p 8000:8000 rabtech-ml-api
```

Then open `http://127.0.0.1:8000/docs`.

## Architecture

```text
Client
   |
   | JSON POST /predict
   v
FastAPI REST API
   |
   | Pydantic validation
   v
Feature Vector
   |
   v
Serialized Champion Model
   |
   v
Prediction + Class Probabilities
   |
   v
JSON Response
```

## Production considerations

For a larger production deployment, this service can be extended with:

- Authentication and authorization
- HTTPS/TLS termination
- Structured logging and request IDs
- Model versioning and registry integration
- Prometheus/Grafana monitoring
- CI/CD pipeline
- Cloud deployment
- Input drift and model-performance monitoring

## Implementation note

The repository is intentionally self-contained so that the API, serialized model, tests, Dockerfile, and documentation can be submitted as one public GitHub repository.

**Important:** If RabTech requires the exact champion model from an earlier internship task, replace `model/champion_model.joblib` and update `model/metadata.json` to match that earlier model's feature schema before final submission. The API structure and tests can remain the same if the input schema is unchanged.

## Internship deliverable checklist

- [x] FastAPI `/predict` endpoint
- [x] JSON payload validation
- [x] Prediction probabilities
- [x] Serialized trained model
- [x] Pinned dependencies
- [x] Dockerfile
- [x] Unit tests
- [x] README architecture documentation
