from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check():
    """Verifica que la API esté funcionando."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_data():
    """Verifica que la API acepte datos válidos."""
    payload = {
        "Fresh": 1000,
        "Milk": 500,
        "Grocery": 800,
        "Frozen": 300,
        "Detergents_Paper": 200,
        "Delicassen": 100
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "cluster" in result
    assert "model_version" in result
    assert isinstance(result["cluster"], int)
    assert result["cluster"] in [0, 1, 2]
    assert result["model_version"] == "baseline-kmeans"


def test_predict_negative_value():
    """Verifica que la API rechace gastos negativos."""
    payload = {
        "Fresh": -500,
        "Milk": 500,
        "Grocery": 800,
        "Frozen": 300,
        "Detergents_Paper": 200,
        "Delicassen": 100
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_required_field():
    """Verifica que la API rechace una variable obligatoria faltante."""
    payload = {
        "Milk": 500,
        "Grocery": 800,
        "Frozen": 300,
        "Detergents_Paper": 200,
        "Delicassen": 100
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_datatype():
    """Verifica que la API rechace tipos de datos incorrectos."""
    payload = {
        "Fresh": "error",
        "Milk": 500,
        "Grocery": 800,
        "Frozen": 300,
        "Detergents_Paper": 200,
        "Delicassen": 100
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422