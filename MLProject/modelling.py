"""
MLProject Modelling Script for CI Workflow
============================================
Script untuk melatih model Heart Disease Classification
dalam konteks MLflow Project (CI pipeline).

Author: I Kadek Rai Pramana
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(data_dir: str = 'heart_disease_preprocessing', test_size: float = 0.2):
    """Load preprocessed data, atau load raw dan preprocess jika perlu."""

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        X_train = train_df.drop('heart_disease', axis=1)
        y_train = train_df['heart_disease']
        X_test = test_df.drop('heart_disease', axis=1)
        y_test = test_df['heart_disease']
    else:
        # Fallback: download and preprocess
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv(url, names=cols, na_values='?')
        df = df.drop_duplicates().dropna()
        df['heart_disease'] = (df['target'] >= 1).astype(int)
        df = df.drop('target', axis=1)

        X = df.drop('heart_disease', axis=1)
        y = df['heart_disease']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(n_estimators: int = 100, max_depth: int = 10, test_size: float = 0.2):
    """Train model dan log ke MLflow."""

    X_train, X_test, y_train, y_test = load_and_prepare_data(test_size=test_size)

    # Deteksi apakah running via 'mlflow run' (env var MLFLOW_RUN_ID di-set)
    mlflow_run_id = os.environ.get('MLFLOW_RUN_ID')

    if mlflow_run_id:
        # Running via mlflow run - langsung log ke run yang sudah dibuat
        run_ctx = mlflow.start_run(run_id=mlflow_run_id)
    else:
        # Running standalone - buat experiment dan run baru
        mlflow.set_experiment("heart-disease-ci")
        run_ctx = mlflow.start_run(run_name="ci_random_forest")

    with run_ctx:
        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, pred_proba)

        # Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("test_size", test_size)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Artifacts
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Has Disease'],
                    yticklabels=['No Disease', 'Has Disease'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)

        # Classification report
        report = classification_report(y_test, predictions, output_dict=True)
        report_path = 'classification_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\nResults:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")

        # Cleanup
        for f_path in [cm_path, report_path]:
            if os.path.exists(f_path):
                os.remove(f_path)

        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    train_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size
    )
