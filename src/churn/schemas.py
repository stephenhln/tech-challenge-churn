"""Contratos de entrada e saída da API (Pydantic v2).

Os tipos ``Literal`` reproduzem exatamente as categorias vistas no treino, de
modo que a validação acontece antes de o dado chegar ao modelo — erro 422 com
mensagem clara em vez de predição silenciosamente errada.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]


class CustomerFeatures(BaseModel):
    """Dados de um cliente da operadora."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "MultipleLines": "No phone service",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": 29.85,
                }
            ]
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 se o cliente tem 65 anos ou mais")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Meses de relacionamento")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)


class BatchPredictionRequest(BaseModel):
    """Lote de clientes para escoragem em massa."""

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Resultado da inferência para um cliente."""

    churn_probability: float = Field(description="Probabilidade estimada de churn (0-1)")
    churn_prediction: int = Field(description="1 = risco de cancelamento, 0 = permanece")
    risk_band: Literal["baixo", "medio", "alto"]
    threshold: float = Field(description="Limiar de decisão aplicado")
    model_name: str


class BatchPredictionResponse(BaseModel):
    """Resultado da inferência em lote."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Status de saúde do serviço."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    api_version: str


class ModelInfoResponse(BaseModel):
    """Metadados do modelo em produção."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    trained_at: str | None = None
    decision_threshold: float
    metrics: dict = Field(default_factory=dict)
    features: dict = Field(default_factory=dict)
