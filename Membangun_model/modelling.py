"""
Modelling Script (Baseline) with MLflow Autolog
Nama: Mohamad Dwi Rezky Desyafa
Dataset: AI4I 2020 Predictive Maintenance Dataset

Script ini melatih model RandomForestClassifier menggunakan MLflow autolog.
Jalankan dari folder Membangun_model:
    python modelling.py

MLflow UI:
    mlflow ui --host 127.0.0.1 --port 5000
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
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
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_and_evaluate():
    """Melatih model baseline dan mengevaluasi dengan MLflow autolog."""
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Set MLflow experiment
    mlflow.set_experiment("predictive-maintenance-baseline")
    
    # Aktifkan autolog
    mlflow.sklearn.autolog()
    
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL")
    print("Model: RandomForestClassifier")
    print("Autolog: Enabled")
    print("=" * 60)
    
    with mlflow.start_run(run_name="baseline-random-forest"):
        # Inisialisasi model baseline
        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )
        
        # Training
        print("\nTraining model...")
        model.fit(X_train, y_train)
        print("Training completed!")
        
        # Prediksi
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Print hasil metrik ke terminal
        print("\n" + "-" * 40)
        print("EVALUATION METRICS:")
        print("-" * 40)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print("-" * 40)
        
        print("\nRun logged to MLflow successfully!")
        print("To view results, run:")
        print("  mlflow ui --host 127.0.0.1 --port 5000")
        print("Then open http://127.0.0.1:5000 in your browser.")
    


if __name__ == "__main__":
    train_and_evaluate()
