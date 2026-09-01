"""Configurações centrais do projeto.

Concentra caminhos, seed e definição de colunas em um único lugar para
garantir que notebooks, scripts de treino, testes e API compartilhem
exatamente a mesma configuração.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Fixa as seeds globais usadas pelo projeto."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

RAW_DATA_PATH: Path = RAW_DATA_DIR / "telco_customer_churn.csv"
MODEL_PATH: Path = MODELS_DIR / "champion_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
EXPERIMENTS_PATH: Path = REPORTS_DIR / "experimentos.csv"

# URL pública do dataset (IBM Telco Customer Churn) usada como fallback
DATA_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# ---------------------------------------------------------------------------
# Esquema dos dados
# ---------------------------------------------------------------------------
TARGET_COL: str = "Churn"
ID_COL: str = "customerID"

NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Split e validação
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Métrica de negócio
# ---------------------------------------------------------------------------
# Custo estimado de perder um cliente (receita anual média perdida) e custo de
# acionar a retenção (desconto/brinde/call center) por cliente contatado.
COST_FALSE_NEGATIVE: float = 1000.0  # cliente que evadiu e não foi abordado
COST_FALSE_POSITIVE: float = 150.0  # incentivo dado a quem não iria evadir
BENEFIT_TRUE_POSITIVE: float = 850.0  # receita retida líquida do incentivo


def ensure_dirs() -> None:
    """Cria os diretórios de saída caso ainda não existam."""
    for path in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
