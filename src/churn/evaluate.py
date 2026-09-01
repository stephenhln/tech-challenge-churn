"""Métricas técnicas e de negócio.

A métrica técnica principal é o **F1 da classe positiva** (churn), porque a
base é desbalanceada (~26,5% de churn) e acurácia premiaria um modelo que
nunca prevê evasão. Como métrica secundária usamos **ROC-AUC** (qualidade do
ranqueamento, independente do limiar).

A métrica de negócio é o **lucro líquido esperado da campanha de retenção**,
calculado a partir da matriz de confusão e dos custos definidos em ``config``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    BENEFIT_TRUE_POSITIVE,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)


@dataclass
class ClassificationMetrics:
    """Conjunto de métricas de um modelo em um conjunto de avaliação."""

    threshold: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    pr_auc: float
    accuracy: float
    business_value: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def business_value(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Valor líquido (R$) da campanha de retenção para uma dada predição.

    - Verdadeiro positivo: cliente em risco abordado e retido -> +benefício.
    - Falso positivo: incentivo dado a quem ficaria de qualquer forma -> -custo.
    - Falso negativo: cliente perdido sem nenhuma ação -> -custo (receita perdida).
    - Verdadeiro negativo: neutro.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(
        tp * BENEFIT_TRUE_POSITIVE
        - fp * COST_FALSE_POSITIVE
        - fn * COST_FALSE_NEGATIVE
    )


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Calcula todas as métricas para um limiar de decisão."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    return ClassificationMetrics(
        threshold=float(threshold),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        accuracy=float((y_pred == y_true).mean()),
        business_value=business_value(y_true, y_pred),
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Busca em grade o limiar que maximiza ``f1`` ou ``business_value``.

    Returns
    -------
    (melhor_limiar, melhor_valor)
    """
    if metric not in {"f1", "business_value"}:
        raise ValueError("metric deve ser 'f1' ou 'business_value'")

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    best_threshold, best_value = 0.5, -np.inf
    for threshold in grid:
        y_pred = (y_proba >= threshold).astype(int)
        value = (
            f1_score(y_true, y_pred, zero_division=0)
            if metric == "f1"
            else business_value(y_true, y_pred)
        )
        if value > best_value:
            best_threshold, best_value = float(threshold), float(value)

    return best_threshold, best_value


def confusion_dict(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    """Matriz de confusão em formato de dicionário, pronta para logar/serializar."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
