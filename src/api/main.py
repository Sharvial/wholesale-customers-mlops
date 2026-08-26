from fastapi import FastAPI

app = FastAPI(
    title="Wholesale Customers ML API",
    description="API para predicción de segmentos de clientes",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}