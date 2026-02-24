import pytest
from fastapi.testclient import TestClient
from src.api.app import app

@pytest.fixture
def client():
    return TestClient(app)


def test_predict_happy_path(client):
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "model" in data


def test_predict_empty_input(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_batch_prediction(client):
    response = client.post(
        "/predict/batch",
        json={"texts": ["I love this", "I hate this"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
