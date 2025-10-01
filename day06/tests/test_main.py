import pytest
from fastapi.testclient import TestClient
from day06.main import app, predict_function
from unittest.mock import patch

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
    assert response.json() == ["model_a", "model_b"]

def test_predict_invalid_model():
    response = client.post("/predict/invalid_model", json={"data": [1,2,3]})
    assert response.status_code == 400
    assert response.json()["detail"] == "Model not found"

def test_predict_invalid_payload():
    response = client.post("/predict/model_a", json={"invalid": "data"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid input format"

def test_predict_valid_model_with_mock():
    with patch("day06.main.predict_function") as mock_predict:
        mock_predict.return_value = {"prediction": 123}
        response = client.post("/predict/model_a", json={"data": [1,2,3]})
        assert response.status_code == 200
        assert response.json() == {"prediction": 123}
        mock_predict.assert_called_once_with("model_a", [1,2,3])
