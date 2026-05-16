from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Verify the API boots and the model loads properly."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predictive_engine():
    """Verify the ML pipeline correctly processes valid customer data."""
    payload = {
        "age": 34,
        "annual_income": 45000.0,
        "credit_score": 620,
        "debt_to_income_ratio": 0.28,
        "missed_payments_last_2_years": 0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert "risk_classification" in response.json()