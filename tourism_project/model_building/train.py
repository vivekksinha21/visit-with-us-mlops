"""Train + tune a classifier, log with MLflow, persist best model."""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR  = Path(__file__).resolve().parents[1] / "deployment"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train.drop(columns=["ProdTaken"])
    y_train = train["ProdTaken"]
    X_test  = test.drop(columns=["ProdTaken"])
    y_test  = test["ProdTaken"]
    return X_train, X_test, y_train, y_test

def evaluate(model, X, y, prefix=""):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds
    metrics = {
        f"{prefix}accuracy":  accuracy_score(y, preds),
        f"{prefix}precision": precision_score(y, preds, zero_division=0),
        f"{prefix}recall":    recall_score(y, preds, zero_division=0),
        f"{prefix}f1":        f1_score(y, preds, zero_division=0),
        f"{prefix}roc_auc":   roc_auc_score(y, proba),
    }
    return metrics, preds

def main():
    print("Loading train / test splits …")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    candidates = {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
            {
                "n_estimators": [100, 200],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [2, 4],
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            }
        ),
    }

    mlflow.set_experiment("tourism-wellness-package")
    best_score = -1
    best_model = None
    best_name  = None
    best_params = None

    for name, (estimator, param_grid) in candidates.items():
        print(f"\n----- Tuning {name} -----")
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(
                estimator, param_grid,
                scoring="f1", cv=3, n_jobs=-1, verbose=1
            )
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            train_m, _ = evaluate(model, X_train, y_train, prefix="train_")
            test_m,  preds = evaluate(model, X_test,  y_test,  prefix="test_")
            mlflow.log_params(gs.best_params_)
            mlflow.log_params({"model_type": name})
            mlflow.log_metrics({**train_m, **test_m})
            mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
            print("Best params:", gs.best_params_)
            print("Test F1:", round(test_m["test_f1"], 4),
                  "  ROC-AUC:", round(test_m["test_roc_auc"], 4))
            if test_m["test_f1"] > best_score:
                best_score = test_m["test_f1"]
                best_model = model
                best_name  = name
                best_params = gs.best_params_

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_name}  (test F1={best_score:.4f})")
    print(f"Saved → {model_path}")

    metrics_path = MODEL_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"best_model={best_name}\n")
        f.write(f"test_f1={best_score:.4f}\n")
        f.write(f"params={best_params}\n")
    print("Metrics written.")

if __name__ == "__main__":
    main()
