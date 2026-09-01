"""Construção do pré-processamento com pipelines do Scikit-Learn.

Todo o pré-processamento vive dentro de um ``ColumnTransformer`` que é
serializado junto com o estimador. Isso elimina *training/serving skew*: a API
carrega um único artefato que já sabe imputar, escalar e codificar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def _make_onehot_encoder() -> OneHotEncoder:
    """OneHotEncoder compatível com versões novas e antigas do Scikit-Learn."""
    try:  # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compat scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` de pré-processamento.

    - Numéricas: imputação pela mediana + padronização.
      A padronização é essencial para Regressão Logística e MLP e inofensiva
      para modelos baseados em árvores.
    - Categóricas: imputação pela moda + One-Hot com ``handle_unknown='ignore'``
      (categoria nunca vista em produção vira um vetor de zeros em vez de erro).
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que o DataFrame tenha exatamente as colunas esperadas, na ordem.

    Colunas ausentes são criadas com ``NaN`` (e serão imputadas pelo pipeline);
    colunas extras são descartadas. É a função usada pela API para blindar a
    entrada antes de chamar ``predict``.
    """
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_COLUMNS]


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nomes das colunas geradas após o pré-processamento (para explicabilidade)."""
    return list(preprocessor.get_feature_names_out())
