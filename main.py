"""import pickle
from contextlib import asynccontextmanager
from config import MODEL_LOGISTIC, MODEL_RF
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import asyncio
from dotenv import load_dotenv

# Import authentication
from auth import get_api_key

# Load environment
load_dotenv()

ml_models = {}


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["logistic_model"] = load_model(MODEL_LOGISTIC)
    ml_models["rf_model"] = load_model(MODEL_RF)
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


# -----------------------
# Schema with validation
# -----------------------
class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm (0-10)")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm (0-10)")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm (0-10)")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm (0-10)")


# -----------------------
# Background logger
# -----------------------
async def log_prediction(model_name: str, features: list, prediction: list):
    # Simulate slow logging
    await asyncio.sleep(3)
    print(f"[LOG] Model={model_name}, Input={features}, Prediction={prediction}")


# -----------------------
# Routes
# -----------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models", dependencies=[Depends(get_api_key)])
async def list_models():
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}", dependencies=[Depends(get_api_key)])
async def predict(
    model_name: Literal["logistic_model", "rf_model"],
    data: IrisData,
    background_tasks: BackgroundTasks
):
    if model_name not in ml_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")

    model = ml_models[model_name]

    # Simulate async heavy computation
    await asyncio.sleep(2)

    features = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    features_array = np.array([features])

    try:
        prediction = model.predict(features_array).tolist()
        probabilities = model.predict_proba(features_array).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Background task for logging
    background_tasks.add_task(log_prediction, model_name, features, prediction)

    return {
        "model": model_name,
        "input": data.dict(),
        "prediction": prediction,
        "probabilities": probabilities
    }
"""

import pickle
import time
import asyncio
import json
from contextlib import asynccontextmanager
from config import MODEL_LOGISTIC, MODEL_RF
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

API_KEY = os.getenv("API_KEY", "changeme")

ml_models = {}
prediction_cache = {}  # (model_name, tuple(features)) -> result


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["logistic_model"] = load_model(MODEL_LOGISTIC)
    ml_models["rf_model"] = load_model(MODEL_RF)
    yield
    ml_models.clear()
    prediction_cache.clear()


app = FastAPI(lifespan=lifespan)


# -----------------------
# Middleware: Auth
# -----------------------
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    whitelist = {"/", "/health", "/docs", "/openapi.json"}
    if request.url.path not in whitelist:
        api_key = request.headers.get("X-API-Key")
        if api_key != API_KEY:
            return JSONResponse(status_code=403, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


# -----------------------
# Middleware: Timing & Logging
# -----------------------
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    process_time = time.perf_counter() - start_time

    # Add timing header
    response.headers["X-Process-Time"] = str(round(process_time, 4))

    # Structured log
    log = {
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "process_time": round(process_time, 4),
    }
    print(json.dumps(log))

    return response


# -----------------------
# Schema with validation
# -----------------------
class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm (0-10)")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm (0-10)")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm (0-10)")
    petal_width: float = Field(..., gt=0, le=10, description="Petal width in cm (0-10)")


# -----------------------
# Background logger
# -----------------------
async def log_prediction(model_name: str, features: list, prediction: list):
    await asyncio.sleep(2)  # simulate slow log
    print(f"[LOG] Model={model_name}, Input={features}, Prediction={prediction}")


# -----------------------
# Routes
# -----------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(
    model_name: Literal["logistic_model", "rf_model"],
    data: IrisData,
    background_tasks: BackgroundTasks
):
    if model_name not in ml_models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")

    features = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    cache_key = (model_name, tuple(features))

    # Check cache
    if cache_key in prediction_cache:
        return {"model": model_name, "input": data.dict(), "cached": True, **prediction_cache[cache_key]}

    # Simulate heavy computation
    await asyncio.sleep(1)
    model = ml_models[model_name]
    prediction = model.predict([features]).tolist()
    probabilities = model.predict_proba([features]).tolist()

    result = {"prediction": prediction, "probabilities": probabilities}
    prediction_cache[cache_key] = result  # Save to cache

    # Background logging
    background_tasks.add_task(log_prediction, model_name, features, prediction)

    return {"model": model_name, "input": data.dict(), "cached": False, **result}

