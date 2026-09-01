"""Testes da API de inferência (FastAPI + TestClient)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.api.main import app
from churn.config import MODEL_PATH

client = TestClient(app)

CLIENTE_ALTO_RISCO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 79.85,
}

CLIENTE_BAIXO_RISCO = {
    **CLIENTE_ALTO_RISCO,
    "tenure": 68,
    "Contract": "Two year",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 60.0,
    "TotalCharges": 4080.0,
}

modelo_necessario = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Modelo não treinado; rode `python -m churn.train` antes.",
)


def test_health_retorna_200_e_status():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "api_version" in body


@modelo_necessario
def test_health_reporta_modelo_carregado():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"]


@modelo_necessario
def test_predict_retorna_probabilidade_valida():
    response = client.post("/predict", json=CLIENTE_ALTO_RISCO)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_band"] in {"baixo", "medio", "alto"}


@modelo_necessario
def test_predict_diferencia_perfis_de_risco():
    """Contrato mensal + cliente novo deve pontuar acima de contrato de 2 anos."""
    alto = client.post("/predict", json=CLIENTE_ALTO_RISCO).json()
    baixo = client.post("/predict", json=CLIENTE_BAIXO_RISCO).json()
    assert alto["churn_probability"] > baixo["churn_probability"]


@modelo_necessario
def test_predict_batch_retorna_uma_predicao_por_cliente():
    payload = {"customers": [CLIENTE_ALTO_RISCO, CLIENTE_BAIXO_RISCO]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_rejeita_categoria_invalida():
    payload = {**CLIENTE_ALTO_RISCO, "Contract": "Contrato vitalicio"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejeita_campo_ausente():
    payload = {k: v for k, v in CLIENTE_ALTO_RISCO.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejeita_valor_fora_do_intervalo():
    payload = {**CLIENTE_ALTO_RISCO, "tenure": -5}
    assert client.post("/predict", json=payload).status_code == 422


@modelo_necessario
def test_model_info_expoe_metadados():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"]
    assert 0.0 < body["decision_threshold"] < 1.0


def test_openapi_documenta_os_endpoints_obrigatorios():
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]
