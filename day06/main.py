from fastapi import FastAPI, HTTPException

app = FastAPI()

# Dummy list of available models
AVAILABLE_MODELS = ["model_a", "model_b"]

@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/health_check")
def health_check():
    return {"status": "ok"}


@app.get("/list_models")
def list_models():
    return AVAILABLE_MODELS


# This is the function we will patch/mock in tests
def predict_function(model_name: str, data: list):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Model not found")
    # Dummy default behavior (should be mocked in tests anyway)
    return {"prediction": sum(data)}


@app.post("/predict/{model_name}")
def predict(model_name: str, payload: dict):
    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Invalid input format")
    return predict_function(model_name, data)
