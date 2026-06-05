"""
Modelling Script for MLflow Project CI
Nama: Mohamad Dwi Rezky Desyafa
Dataset: AI4I 2020 Predictive Maintenance Dataset

Script ini digunakan dalam MLflow Project untuk workflow CI.
Melatih model RandomForestClassifier, menyimpan artifact secara lokal,
dan melacak eksperimen menggunakan MLflow.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn


def load_preprocessed_data(data_dir="predictive_maintenance_preprocessing"):
    """Memuat dataset hasil preprocessing."""
    print(f"Loading preprocessed data from: {data_dir}")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def main():
    """Training pipeline untuk CI workflow."""
    print("=" * 60)
    print("MLflow Project — Predictive Maintenance Training")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    feature_names = X_train.columns.tolist()
    
    # Aktifkan autolog (MLflow run context is managed by 'mlflow run .')
    mlflow.sklearn.autolog()
    
    # Model
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_estimators=100
    )
    
    # Training
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training completed!")
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrik
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print metrik
    print("\n" + "-" * 40)
    print("EVALUATION METRICS:")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("-" * 40)
    
    # Simpan artifact lokal
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Machine Failure'],
                yticklabels=['Normal', 'Machine Failure'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Machine Failure'])
    report_path = os.path.join(artifacts_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # Save model
    model_path = os.path.join(artifacts_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Log artifacts ke MLflow
    mlflow.log_artifact(metrics_path)
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(model_path)
    
    print("\n" + "=" * 60)
    print("CI TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
