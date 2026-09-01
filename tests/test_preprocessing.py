"""Testes das funções de limpeza e pré-processamento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.config import FEATURE_COLUMNS
from churn.data import clean_data, encode_target, split_features_target
from churn.preprocessing import align_features, build_preprocessor


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """Amostra sintética que reproduz as sujeiras reais do dataset."""
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB", "0002-BBB", "0003-CCC"],
            "gender": ["Female", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 1, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "No", "Yes"],
            "tenure": [1, 34, 34, 0],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "No", "No"],
            "InternetService": ["DSL", "DSL", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "No", "No"],
            "DeviceProtection": ["No", "Yes", "Yes", "No"],
            "TechSupport": ["No", "No", "No", "No"],
            "StreamingTV": ["No", "No", "No", "Yes"],
            "StreamingMovies": ["No", "No", "No", "Yes"],
            "Contract": ["Month-to-month", "One year", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Mailed check",
                "Electronic check",
            ],
            "MonthlyCharges": [29.85, 56.95, 56.95, 70.70],
            # a terceira posição traz o espaço em branco do CSV original
            "TotalCharges": ["29.85", "1889.5", "1889.5", " "],
            "Churn": ["No", "No", "No", "Yes"],
        }
    )


def test_clean_data_converte_total_charges_para_numerico(raw_sample):
    cleaned = clean_data(raw_sample)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0
    # cliente com tenure 0 nunca foi faturado -> imputado com 0.0
    assert cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"].iloc[0] == 0.0


def test_clean_data_remove_duplicatas_e_id(raw_sample):
    cleaned = clean_data(raw_sample)
    assert "customerID" not in cleaned.columns
    assert len(cleaned) == 3  


def test_clean_data_nao_altera_o_dataframe_original(raw_sample):
    antes = raw_sample.copy()
    clean_data(raw_sample)
    pd.testing.assert_frame_equal(raw_sample, antes)


def test_encode_target_mapeia_yes_no_para_1_0():
    resultado = encode_target(pd.Series(["Yes", "No", "Yes"]))
    assert resultado.tolist() == [1, 0, 1]


def test_encode_target_rejeita_valores_invalidos():
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "Talvez"]))


def test_split_features_target_retorna_colunas_esperadas(raw_sample):
    X, y = split_features_target(clean_data(raw_sample))
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_align_features_cria_colunas_faltantes_e_descarta_extras():
    entrada = pd.DataFrame([{"tenure": 5, "coluna_intrusa": "x"}])
    alinhado = align_features(entrada)
    assert list(alinhado.columns) == FEATURE_COLUMNS
    assert "coluna_intrusa" not in alinhado.columns
    assert alinhado["tenure"].iloc[0] == 5
    assert np.isnan(alinhado["MonthlyCharges"].iloc[0])


def test_preprocessor_gera_matriz_numerica_sem_nulos(raw_sample):
    X, _ = split_features_target(clean_data(raw_sample))
    preprocessor = build_preprocessor()
    transformado = preprocessor.fit_transform(X)
    assert transformado.shape[0] == len(X)
    assert transformado.shape[1] > len(FEATURE_COLUMNS)  
    assert not np.isnan(transformado).any()


def test_preprocessor_ignora_categoria_desconhecida_em_producao(raw_sample):
    X, _ = split_features_target(clean_data(raw_sample))
    preprocessor = build_preprocessor().fit(X)

    novo = X.iloc[[0]].copy()
    novo["Contract"] = "Contrato vitalicio"  # categoria nunca vista no treino
    transformado = preprocessor.transform(novo)  # não deve levantar exceção

    assert transformado.shape[1] == preprocessor.transform(X.iloc[[0]]).shape[1]
