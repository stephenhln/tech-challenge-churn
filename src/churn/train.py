"""Pipeline de treino: compara candidatos, elege o campeão e salva o artefato.

Uso::

    python -m churn.train                 # treina tudo e salva o campeão
    python -m churn.train --metric roc_auc
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import __version__
from .config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    EXPERIMENTS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COL,
    ensure_dirs,
    set_seeds,
)
from .data import get_train_test_split, load_clean_data
from .evaluate import compute_metrics, confusion_dict, find_best_threshold
from .models import get_all_models

CV_SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "average_precision": "average_precision",
}


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Roda validação cruzada estratificada para todos os candidatos."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows: list[dict[str, object]] = []

    for name, pipeline in get_all_models(random_state=random_state).items():
        print(f"[train] Validação cruzada: {name} ...", flush=True)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=1,
            return_train_score=False,
        )
        row: dict[str, object] = {"model": name, "fit_time_s": float(np.mean(scores["fit_time"]))}
        for metric in CV_SCORING:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values))
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_on_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Treina cada candidato no treino completo e avalia no hold-out."""
    rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}

    for name, pipeline in get_all_models(random_state=random_state).items():
        print(f"[train] Treino final + hold-out: {name} ...", flush=True)
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]

        default = compute_metrics(y_test, proba, threshold=0.5)
        best_threshold, _ = find_best_threshold(y_test, proba, metric="f1")
        tuned = compute_metrics(y_test, proba, threshold=best_threshold)

        rows.append(
            {
                "model": name,
                "test_f1": default.f1,
                "test_precision": default.precision,
                "test_recall": default.recall,
                "test_roc_auc": default.roc_auc,
                "test_pr_auc": default.pr_auc,
                "test_accuracy": default.accuracy,
                "test_business_value": default.business_value,
                "best_threshold": tuned.threshold,
                "test_f1_tuned": tuned.f1,
                "test_business_value_tuned": tuned.business_value,
            }
        )
        fitted[name] = pipeline

    return pd.DataFrame(rows), fitted


def select_champion(results: pd.DataFrame, metric: str = "f1") -> str:
    """Elege o campeão pela média da validação cruzada (evita overfit no teste)."""
    column = f"cv_{metric}_mean"
    if column not in results.columns:
        raise KeyError(f"Coluna '{column}' ausente na tabela de resultados.")
    return str(results.loc[results[column].idxmax(), "model"])


def train(metric: str = "f1", random_state: int = RANDOM_SEED) -> dict[str, object]:
    """Executa o fluxo completo de treino e persiste os artefatos."""
    set_seeds(random_state)
    ensure_dirs()

    print("[train] Carregando e limpando dados ...")
    df = load_clean_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df, random_state=random_state)
    print(f"[train] Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"[train] Taxa de churn (treino): {y_train.mean():.4f}")

    cv_results = cross_validate_models(X_train, y_train, random_state=random_state)
    holdout_results, fitted = evaluate_on_holdout(
        X_train, y_train, X_test, y_test, random_state=random_state
    )

    comparison = cv_results.merge(holdout_results, on="model")
    champion_name = select_champion(comparison, metric=metric)
    comparison["champion"] = comparison["model"] == champion_name
    comparison = comparison.sort_values(f"cv_{metric}_mean", ascending=False)
    comparison.to_csv(EXPERIMENTS_PATH, index=False)
    print(f"[train] Tabela comparativa salva em {EXPERIMENTS_PATH}")

    champion = fitted[champion_name]
    proba = champion.predict_proba(X_test)[:, 1]
    best_threshold, _ = find_best_threshold(y_test, proba, metric="f1")
    metrics_default = compute_metrics(y_test, proba, threshold=0.5)
    metrics_tuned = compute_metrics(y_test, proba, threshold=best_threshold)

    joblib.dump(champion, MODEL_PATH)
    print(f"[train] Modelo campeão ({champion_name}) salvo em {MODEL_PATH}")

    metadata = {
        "model_name": champion_name,
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": random_state,
        "selection_metric": metric,
        "decision_threshold": best_threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_train": float(y_train.mean()),
        "target": TARGET_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics_threshold_0_5": metrics_default.to_dict(),
        "metrics_tuned_threshold": metrics_tuned.to_dict(),
        "confusion_matrix_tuned": confusion_dict(y_test, proba, best_threshold),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[train] Metadados salvos em {MODEL_METADATA_PATH}")

    print("\n=== Comparação de modelos ===")
    cols = ["model", f"cv_{metric}_mean", f"cv_{metric}_std", "test_f1", "test_roc_auc", "champion"]
    print(comparison[cols].to_string(index=False))

    return {"champion": champion_name, "comparison": comparison, "metadata": metadata}


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Treina e seleciona o modelo de churn")
    parser.add_argument(
        "--metric",
        default="f1",
        choices=sorted(CV_SCORING),
        help="métrica de seleção do campeão (padrão: f1)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    train(metric=args.metric, random_state=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
