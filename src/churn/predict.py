"""Serviço de inferência: carrega o artefato e produz predições.

Isola a API de detalhes do modelo. O carregamento é *lazy* e cacheado para que
o artefato seja lido do disco uma única vez por processo.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODEL_METADATA_PATH, MODEL_PATH
from .preprocessing import align_features

DEFAULT_THRESHOLD = 0.5
RISK_BANDS = ((0.30, "baixo"), (0.60, "medio"))  # limites superiores exclusivos


class ModelNotFoundError(RuntimeError):
    """Levantada quando o artefato do modelo não existe no disco."""


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> Any:
    """Carrega o pipeline campeão (com cache em memória)."""
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Modelo não encontrado em '{model_path}'. Rode `python -m churn.train`."
        )
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_metadata(path: str | None = None) -> dict[str, Any]:
    """Carrega os metadados do treino (nome do modelo, limiar, métricas)."""
    metadata_path = Path(path) if path else MODEL_METADATA_PATH
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def get_threshold() -> float:
    """Limiar de decisão calibrado no treino (fallback: 0.5)."""
    return float(load_metadata().get("decision_threshold", DEFAULT_THRESHOLD))


def risk_band(probability: float) -> str:
    """Traduz a probabilidade em uma faixa de risco acionável pelo time de CRM."""
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return "alto"


def predict_proba(records: Iterable[dict[str, Any]] | pd.DataFrame) -> list[float]:
    """Probabilidade de churn para um ou mais clientes."""
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty:
        return []
    model = load_model()
    X = align_features(df)
    return [float(p) for p in model.predict_proba(X)[:, 1]]


def predict(
    records: Iterable[dict[str, Any]] | pd.DataFrame,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Predição completa: probabilidade, classe, faixa de risco e limiar usado."""
    threshold = get_threshold() if threshold is None else float(threshold)
    probabilities = predict_proba(records)
    return [
        {
            "churn_probability": round(p, 6),
            "churn_prediction": int(p >= threshold),
            "risk_band": risk_band(p),
            "threshold": threshold,
        }
        for p in probabilities
    ]
