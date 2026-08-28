from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# ============================================================
# CARGA DEL PIPELINE Y MODELO
# ============================================================

pipeline = joblib.load("models/feature_pipeline.pkl")
model = joblib.load("models/baseline_kmeans.pkl")


# ============================================================
# CONFIGURACIÓN DE LA API
# ============================================================

app = FastAPI(
    title="Wholesale Customers ML API",
    description="API para segmentación de clientes mediante K-Means",
    version="1.0.0"
)


# ============================================================
# MODELO DE ENTRADA
# ============================================================

class CustomerData(BaseModel):
    Fresh: float = Field(ge=0, description="Gasto anual en productos Fresh")
    Milk: float = Field(ge=0, description="Gasto anual en productos Milk")
    Grocery: float = Field(ge=0, description="Gasto anual en productos Grocery")
    Frozen: float = Field(ge=0, description="Gasto anual en productos Frozen")
    Detergents_Paper: float = Field(
        ge=0,
        description="Gasto anual en detergentes y papel"
    )
    Delicassen: float = Field(
        ge=0,
        description="Gasto anual en productos Delicassen"
    )


# ============================================================
# MODELO DE RESPUESTA
# ============================================================

class PredictionResponse(BaseModel):
    cluster: int
    model_version: str


# ============================================================
# ENDPOINT DE SALUD
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# ============================================================
# ENDPOINT DE PREDICCIÓN
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):

    # Convertir los datos recibidos en DataFrame
    data = pd.DataFrame([customer.model_dump()])

    # Aplicar el mismo pipeline utilizado durante el entrenamiento
    data_transformed = pipeline.transform(data)

    # Realizar predicción
    cluster = model.predict(data_transformed)

    return {
        "cluster": int(cluster[0]),
        "model_version": "baseline-kmeans"
    }