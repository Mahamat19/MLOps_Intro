"""from fastapi import FastAPI
from contextlib import asynccontextmanager
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Sample model data (for demonstration)
MODELS = [
    {"id": 1, "name": "Linear Regression", "type": "Regression"},
    {"id": 2, "name": "Random Forest", "type": "Classification"},
    {"id": 3, "name": "Logistic Regression", "type": "Classification"},
]

# Lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("App is starting up... (connect to DB, load model, etc.)")
    yield
    # Shutdown logic
    print("App is shutting down... (close DB, cleanup resources)")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI with lifespan!"}

@app.get("/models")
async def get_models():
    return {"models": MODELS}
"""

import os

#from dotenv import load_dotenv

#load_dotenv()
#print(os.getenv("MODEL_LOGISTIC"))
#print(os.getenv("MODEL_RF"))
MODEL_LOGISTIC= os.getenv("MODEL_LOGISTIC")
MODEL_RF = os.getenv("MODEL_RF")