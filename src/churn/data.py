"""Carregamento e limpeza do dataset de churn.

Funções puras e testáveis: recebem/retornam DataFrames e não dependem de
estado global além da configuração.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    FEATURE_COLUMNS,
    ID_COL,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Lê o CSV bruto do disco.

    Parameters
    ----------
    path:
        Caminho do arquivo. Se omitido, usa ``config.RAW_DATA_PATH``.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir (com instrução de como baixá-lo).
    """
    path = Path(path) if path is not None else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{path}'. "
            "Rode `python -m churn.data --download` para baixá-lo."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza determinística do dataset bruto.

    Passos:
      1. remove espaços em branco nas colunas de texto;
      2. converte ``TotalCharges`` para numérico (11 linhas vêm em branco para
         clientes com ``tenure == 0``, que ainda não foram faturados);
      3. imputa ``TotalCharges`` ausente com 0.0 (regra de negócio: cliente
         novo, nada faturado ainda);
      4. remove linhas duplicadas e a coluna de ID.

    A função **não** altera o DataFrame de entrada.
    """
    df = df.copy()

    text_cols = [
        col for col in df.columns if pd.api.types.is_string_dtype(df[col])
    ]
    for col in text_cols:
        df[col] = df[col].astype("object").str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL]).drop(columns=[ID_COL])
    else:
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def encode_target(series: pd.Series) -> pd.Series:
    """Converte o alvo ``Yes``/``No`` em 1/0 (int).

    Aceita valores já numéricos ou booleanos sem quebrar.
    """
    if not pd.api.types.is_numeric_dtype(series):
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "True": 1, "False": 0}
        encoded = series.astype("object").map(mapping)
        if encoded.isna().any():
            invalid = sorted(set(series[encoded.isna()].astype(str)))
            raise ValueError(f"Valores inesperados no alvo: {invalid}")
        return encoded.astype(int)
    return series.astype(int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features do contrato) e y (alvo codificado)."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Coluna alvo '{TARGET_COL}' ausente no DataFrame.")
    y = encode_target(df[TARGET_COL])
    X = df[FEATURE_COLUMNS].copy()
    return X, y


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado e reprodutível em treino/teste."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_clean_data(path: str | Path | None = None) -> pd.DataFrame:
    """Atalho: carrega o CSV bruto e devolve já limpo."""
    return clean_data(load_raw_data(path))


def download_dataset(path: str | Path | None = None) -> Path:
    """Baixa o dataset público da IBM caso ainda não exista localmente."""
    import urllib.request

    from .config import DATA_URL

    path = Path(path) if path is not None else RAW_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[data] Dataset já presente em {path}")
        return path
    print(f"[data] Baixando dataset de {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, path)  # noqa: S310
    print(f"[data] Salvo em {path}")
    return path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Utilitários de dados de churn")
    parser.add_argument("--download", action="store_true", help="baixa o CSV bruto")
    args = parser.parse_args()

    if args.download:
        download_dataset()
    else:
        frame = load_clean_data()
        print(frame.shape)
        print(frame.head())
