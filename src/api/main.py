from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import time


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
# MÉTRICAS DEL SISTEMA
# ============================================================

system_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_response_time": 0.0
}


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

    start_time = time.perf_counter()

    system_metrics["total_requests"] += 1

    try:
        # Convertir los datos recibidos en DataFrame
        data = pd.DataFrame([customer.model_dump()])

        # Aplicar el mismo pipeline utilizado durante el entrenamiento
        data_transformed = pipeline.transform(data)

        # Realizar predicción
        cluster = model.predict(data_transformed)

        system_metrics["successful_requests"] += 1

        return {
            "cluster": int(cluster[0]),
            "model_version": "baseline-kmeans"
        }

    except Exception:
        system_metrics["failed_requests"] += 1
        raise

    finally:
        elapsed_time = time.perf_counter() - start_time
        system_metrics["total_response_time"] += elapsed_time


# ============================================================
# ENDPOINT DE MÉTRICAS DEL SISTEMA
# ============================================================

@app.get("/metrics")
def get_metrics():

    total_requests = system_metrics["total_requests"]

    if total_requests > 0:
        average_response_time = (
            system_metrics["total_response_time"] / total_requests
        )
    else:
        average_response_time = 0.0

    return {
        "total_requests": total_requests,
        "successful_requests": system_metrics["successful_requests"],
        "failed_requests": system_metrics["failed_requests"],
        "average_response_time_seconds": round(
            average_response_time, 6
        )
    }