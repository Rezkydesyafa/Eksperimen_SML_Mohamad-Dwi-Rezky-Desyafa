"""
Modelling Tuning Script with Manual MLflow Logging
Nama: Mohamad Dwi Rezky Desyafa
Dataset: AI4I 2020 Predictive Maintenance Dataset

Script ini melatih model RandomForestClassifier dengan GridSearchCV
dan menggunakan manual logging MLflow (tanpa autolog).
Jalankan dari folder Membangun_model:
    python modelling_tuning.py

MLflow UI:
    mlflow ui --host 127.0.0.1 --port 5000
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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


def create_artifacts(model, X_test, y_test, y_pred, feature_names, artifacts_dir="artifacts"):
    """Membuat artifact untuk di-log ke MLflow."""
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    print("Creating confusion matrix plot...")
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
    print(f"  Saved: {cm_path}")
    
    # 2. Classification Report
    print("Creating classification report...")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Machine Failure'])
    report_path = os.path.join(artifacts_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  Saved: {report_path}")
    
    # 3. Feature Importance
    print("Creating feature importance plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    fi_path = os.path.join(artifacts_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"  Saved: {fi_path}")
    
    # 4. Best Model (pkl)
    print("Saving best model...")
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"  Saved: {model_path}")
    
    return artifacts_dir


def save_preprocessor_for_inference(artifacts_dir="artifacts"):
    """
    Build and save a preprocessor (ColumnTransformer) for the inference API.
    Reads the raw dataset, fits the preprocessor on training split, and saves it.
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    # Try to load raw data to fit preprocessor
    raw_paths = [
        os.path.join("..", "..", "Eksperimen_SML_Mohamad-Dwi-Rezky-Desyafa",
                      "predictive_maintenance_raw", "predictive_maintenance.csv"),
        os.path.join("..", "..", "ai4i2020.csv"),
        "predictive_maintenance.csv"
    ]
    
    raw_df = None
    for path in raw_paths:
        if os.path.exists(path):
            raw_df = pd.read_csv(path)
            print(f"  Raw data loaded from: {path}")
            break
    
    if raw_df is None:
        print("  WARNING: Raw data not found. Preprocessor.pkl not created.")
        print("  The inference API will use manual preprocessing fallback.")
        return
    
    # Clean data
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_clean = raw_df.drop(columns=columns_to_drop)
    
    # Separate features and target
    X = df_clean.drop(columns=['Machine failure'])
    y = df_clean['Machine failure']
    
    # Split (same params as training)
    X_train_raw, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build preprocessor
    numerical_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    categorical_features = ['Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Fit on training data
    preprocessor.fit(X_train_raw)
    
    # Save
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Preprocessor saved to: {preprocessor_path}")


def train_with_tuning():
    """Melatih model dengan hyperparameter tuning dan manual MLflow logging."""
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    feature_names = X_train.columns.tolist()
    
    # Set MLflow experiment
    mlflow.set_experiment("predictive-maintenance-tuning")
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("Model: RandomForestClassifier")
    print("Method: GridSearchCV")
    print("Manual Logging: Enabled")
    print("=" * 60)
    
    # Parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"]
    }
    
    print(f"\nParameter grid: {param_grid}")
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    print(f"Total combinations: {total_combinations}")
    
    # Model base
    base_model = RandomForestClassifier(random_state=42)
    
    # GridSearchCV
    print("\nRunning GridSearchCV (this may take a few minutes)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # Prediksi dengan best model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrik
    print("\n" + "-" * 40)
    print("EVALUATION METRICS (Best Model):")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("-" * 40)
    
    # Buat artifact
    artifacts_dir = create_artifacts(best_model, X_test, y_test, y_pred, feature_names)
    
    # Manual logging ke MLflow
    print("\n--- Logging to MLflow ---")
    with mlflow.start_run(run_name="tuned-random-forest"):
        # Log parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("cv_folds", 5)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("best_cv_f1_score", grid_search.best_score_)
        
        # Log artifacts
        mlflow.log_artifact(os.path.join(artifacts_dir, "confusion_matrix.png"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "classification_report.txt"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "feature_importance.png"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "best_model.pkl"))
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        print("All parameters, metrics, and artifacts logged to MLflow!")
    
    # Save feature columns for inference
    feature_columns_path = os.path.join(artifacts_dir, "feature_columns.json")
    with open(feature_columns_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature columns saved to: {feature_columns_path}")
    
    # Build and save preprocessor for inference API
    print("\n--- Building Preprocessor for Inference ---")
    save_preprocessor_for_inference(artifacts_dir)
    
    print("\n" + "=" * 60)
    print("TUNING COMPLETED SUCCESSFULLY!")
    print("To view results, run:")
    print("  mlflow ui --host 127.0.0.1 --port 5000")
    print("Then open http://127.0.0.1:5000 in your browser.")
    print("=" * 60)


if __name__ == "__main__":
    train_with_tuning()
