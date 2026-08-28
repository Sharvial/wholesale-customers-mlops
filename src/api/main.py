from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# Cargar pipeline y modelo
pipeline = joblib.load("models/feature_pipeline.pkl")
model = joblib.load("models/baseline_kmeans.pkl")


# Crear aplicación FastAPI
app = FastAPI(
    title="Wholesale Customers ML API",
    description="API para predicción de segmentos de clientes",
    version="1.0.0"
)


# Datos que recibirá la API
class CustomerData(BaseModel):
    Fresh: float
    Milk: float
    Grocery: float
    Frozen: float
    Detergents_Paper: float
    Delicassen: float


# Endpoint de salud
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Endpoint de predicción
@app.post("/predict")
def predict(customer: CustomerData):

    # Convertir los datos recibidos en DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Aplicar las transformaciones del pipeline
    data_transformed = pipeline.transform(data)

    # Realizar la predicción
    cluster = model.predict(data_transformed)

    return {
        "cluster": int(cluster[0])
    }