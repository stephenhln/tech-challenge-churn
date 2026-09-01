"""Testes dos modelos, das métricas e da reprodutibilidade."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_SEED
from churn.evaluate import business_value, compute_metrics, find_best_threshold
from churn.models import MODEL_REGISTRY, get_all_models, get_model
from churn.predict import risk_band


@pytest.fixture
def dados_sinteticos() -> tuple[pd.DataFrame, pd.Series]:
    """Gera uma base pequena com o mesmo esquema do dataset real."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 200
    dados = {
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n).round(2),
        "TotalCharges": rng.uniform(0, 8000, n).round(2),
        "SeniorCitizen": rng.integers(0, 2, n),
    }
    categorias = {
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }
    for coluna, valores in categorias.items():
        dados[coluna] = rng.choice(valores, n)

    X = pd.DataFrame(dados)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    # alvo com sinal real: contrato mensal e pouco tempo de casa elevam o risco
    logito = (
        -0.5
        - 0.04 * X["tenure"]
        + 1.2 * (X["Contract"] == "Month-to-month").astype(int)
        + 0.01 * X["MonthlyCharges"]
    )
    prob = 1 / (1 + np.exp(-logito))
    y = pd.Series(rng.binomial(1, prob), name="Churn")
    return X, y


def test_registry_contem_as_tres_familias_exigidas():
    assert {"logistic_regression", "random_forest", "mlp"} <= set(MODEL_REGISTRY)


def test_get_model_levanta_erro_para_nome_desconhecido():
    with pytest.raises(KeyError):
        get_model("xgboost_magico")


@pytest.mark.parametrize("nome", sorted(MODEL_REGISTRY))
def test_modelos_treinam_e_preveem_probabilidades(nome, dados_sinteticos):
    X, y = dados_sinteticos
    modelo = get_model(nome)
    assert isinstance(modelo, Pipeline)
    modelo.fit(X, y)
    proba = modelo.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_treino_e_reprodutivel_com_a_mesma_seed(dados_sinteticos):
    X, y = dados_sinteticos
    p1 = get_model("random_forest", random_state=RANDOM_SEED).fit(X, y).predict_proba(X)[:, 1]
    p2 = get_model("random_forest", random_state=RANDOM_SEED).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_allclose(p1, p2)


def test_get_all_models_instancia_pipelines_independentes():
    modelos = get_all_models()
    assert len(modelos) == len(MODEL_REGISTRY)
    assert all(isinstance(m, Pipeline) for m in modelos.values())


def test_compute_metrics_para_classificador_perfeito():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.10, 0.90, 0.99])
    metricas = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metricas.f1 == pytest.approx(1.0)
    assert metricas.roc_auc == pytest.approx(1.0)
    assert metricas.accuracy == pytest.approx(1.0)


def test_business_value_penaliza_falsos_negativos():
    y_true = np.array([1, 1, 0, 0])
    acertando = business_value(y_true, np.array([1, 1, 0, 0]))
    ignorando = business_value(y_true, np.array([0, 0, 0, 0]))
    assert acertando > ignorando


def test_find_best_threshold_encontra_limiar_valido():
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = rng.binomial(1, 0.3, 500)
    y_proba = np.clip(y_true * 0.4 + rng.uniform(0, 0.6, 500), 0, 1)
    limiar, valor = find_best_threshold(y_true, y_proba, metric="f1")
    assert 0.0 < limiar < 1.0
    assert valor >= compute_metrics(y_true, y_proba, threshold=0.5).f1


@pytest.mark.parametrize(
    ("probabilidade", "faixa"),
    [(0.05, "baixo"), (0.29, "baixo"), (0.45, "medio"), (0.75, "alto"), (0.99, "alto")],
)
def test_risk_band_classifica_faixas(probabilidade, faixa):
    assert risk_band(probabilidade) == faixa
