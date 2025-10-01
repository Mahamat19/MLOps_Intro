import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


def test_health_check():
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models():
    response = client.get("/list_models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert "model_a" in models


def test_predict_invalid_model():
    response = client.post("/predict/invalid_model", json={"data": [1, 2, 3]})
    assert response.status_code == 400
    assert "Model not found" in response.json()["detail"]


def test_predict_valid_model(mocker):
    # Patch predict_function to avoid real computation
    mocker.patch("main.predict_function", return_value={"prediction": 42})

    response = client.post("/predict/model_a", json={"data": [1, 2, 3]})
    assert response.status_code == 200
    assert response.json() == {"prediction": 42}
