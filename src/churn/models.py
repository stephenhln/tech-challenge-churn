"""Catálogo de modelos candidatos.

Cada candidato é um ``Pipeline`` completo (pré-processamento + estimador),
o que permite validação cruzada honesta: o *fit* do scaler/encoder acontece
dentro de cada fold, sem vazamento de dados.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED
from .preprocessing import build_preprocessor


def build_logistic_regression(random_state: int = RANDOM_SEED) -> Pipeline:
    """Baseline linear — interpretável, rápido e difícil de bater em dados tabulares."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    C=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest(random_state: int = RANDOM_SEED) -> Pipeline:
    """Ensemble de árvores — captura interações não lineares sem *feature engineering*."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_mlp(random_state: int = RANDOM_SEED) -> Pipeline:
    """Rede neural rasa (MLPClassifier) — exige features padronizadas."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    batch_size=64,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=15,
                    validation_fraction=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )


MODEL_REGISTRY: dict[str, Callable[[], Pipeline]] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "mlp": build_mlp,
}


def get_model(name: str, random_state: int = RANDOM_SEED) -> Pipeline:
    """Instancia um candidato pelo nome registrado."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Modelo '{name}' desconhecido. Disponíveis: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](random_state=random_state)


def get_all_models(random_state: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Instancia todos os candidatos do registro."""
    return {name: factory(random_state=random_state) for name, factory in MODEL_REGISTRY.items()}
