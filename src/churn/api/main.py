"""API de inferência de churn (FastAPI).

Endpoints
---------
GET  /health        — verificação de saúde e disponibilidade do modelo
POST /predict       — escoragem de um cliente
POST /predict/batch — escoragem de até 1000 clientes
GET  /model/info    — metadados e métricas do modelo em produção

Execução local::

    uvicorn churn.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..predict import (
    ModelNotFoundError,
    get_threshold,
    load_metadata,
    load_model,
)
from ..predict import (
    predict as run_prediction,
)
from ..schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo no startup (falha rápido e evita latência na 1ª chamada)."""
    try:
        load_model()
        metadata = load_metadata()
        logger.info(
            "Modelo carregado: %s (limiar=%.2f)",
            metadata.get("model_name", "desconhecido"),
            get_threshold(),
        )
    except ModelNotFoundError as exc:
        logger.error("Falha ao carregar o modelo: %s", exc)
    yield
    logger.info("Encerrando a API de churn")


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "API de inferência do modelo de propensão a churn de clientes de "
        "telecomunicações. Tech Challenge — Fase 1 (POS Tech)."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_is_available() -> bool:
    try:
        load_model()
        return True
    except ModelNotFoundError:
        return False


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": "Churn Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Retorna 200 sempre que o processo está de pé.

    O campo ``status`` fica ``degraded`` se o artefato do modelo não puder ser
    carregado, útil para diferenciar "processo vivo" de "pronto para servir".
    """
    loaded = _model_is_available()
    metadata = load_metadata() if loaded else {}
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=metadata.get("model_name"),
        api_version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_one(customer: CustomerFeatures) -> PredictionResponse:
    """Calcula a propensão de churn de um cliente."""
    try:
        result = run_prediction([customer.model_dump()])[0]
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:  # pragma: no cover - rede de segurança
        logger.exception("Erro inesperado na inferência")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar a predição: {exc}",
        ) from exc

    metadata = load_metadata()
    logger.info(
        "predict contract=%s tenure=%s -> p=%.4f",
        customer.Contract,
        customer.tenure,
        result["churn_probability"],
    )
    return PredictionResponse(model_name=metadata.get("model_name", "unknown"), **result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Escoragem em lote (até 1000 clientes por chamada)."""
    try:
        results = run_prediction([c.model_dump() for c in request.customers])
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    model_name = load_metadata().get("model_name", "unknown")
    predictions = [PredictionResponse(model_name=model_name, **r) for r in results]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["infra"])
def model_info() -> ModelInfoResponse:
    """Expõe metadados do modelo servido (versão, limiar, métricas de teste)."""
    metadata = load_metadata()
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadados do modelo indisponíveis. Rode `python -m churn.train`.",
        )
    return ModelInfoResponse(
        model_name=metadata.get("model_name", "unknown"),
        trained_at=metadata.get("trained_at"),
        decision_threshold=get_threshold(),
        metrics=metadata.get("metrics_tuned_threshold", {}),
        features={
            "numeric": metadata.get("numeric_features", []),
            "categorical": metadata.get("categorical_features", []),
        },
    )
