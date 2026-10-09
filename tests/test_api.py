from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid_payload():
    payload = {
        "sepal_length_cm": 5.1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in {"setosa", "versicolor", "virginica"}
    assert body["predicted_class_index"] in {0, 1, 2}
    assert len(body["probabilities"]) == 3
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-5


def test_predict_rejects_invalid_payload():
    payload = {
        "sepal_length_cm": -1,
        "sepal_width_cm": 3.5,
        "petal_length_cm": 1.4,
        "petal_width_cm": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
