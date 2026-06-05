"""
Flask Inference API for Predictive Maintenance Model
Nama: Mohamad Dwi Rezky Desyafa

Endpoint:
  GET  /          - Health check
  POST /predict   - Predict machine failure
  GET  /metrics   - Prometheus metrics

Jalankan:
    python 7.Inference.py
"""

import os
import time
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)

# ============================================================
# Prometheus Metrics
# ============================================================
# 1. Total prediction requests
total_prediction_requests = Counter(
    'total_prediction_requests',
    'Total number of prediction requests received'
)

# 2. Successful prediction requests
successful_prediction_requests = Counter(
    'successful_prediction_requests',
    'Total number of successful prediction requests'
)

# 3. Failed prediction requests
failed_prediction_requests = Counter(
    'failed_prediction_requests',
    'Total number of failed prediction requests'
)

# 4. Prediction latency
prediction_latency_seconds = Histogram(
    'prediction_latency_seconds',
    'Time taken to process a prediction request',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 5. Predicted machine failure total
predicted_machine_failure_total = Counter(
    'predicted_machine_failure_total',
    'Total number of predictions classified as Machine Failure'
)

# 6. Predicted normal machine total
predicted_normal_machine_total = Counter(
    'predicted_normal_machine_total',
    'Total number of predictions classified as Normal'
)

# ============================================================
# Load Model and Preprocessor
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "preprocessor.pkl")
FEATURE_COLUMNS_PATH = os.environ.get("FEATURE_COLUMNS_PATH", "feature_columns.json")

model = None
preprocessor = None
feature_columns = None

# Kolom fitur yang diharapkan dari input JSON (fitur raw sebelum preprocessing)
RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def load_model():
    """Load model, preprocessor, dan feature columns."""
    global model, preprocessor, feature_columns
    
    # Load model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from: {MODEL_PATH}")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please ensure best_model.pkl is in the current directory.")
        print("You can copy it from Membangun_model/artifacts/best_model.pkl")
    
    # Load preprocessor
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"Preprocessor loaded from: {PREPROCESSOR_PATH}")
    else:
        print(f"WARNING: Preprocessor file not found at {PREPROCESSOR_PATH}")
        print("Inference will use manual preprocessing fallback.")
    
    # Load feature columns
    if os.path.exists(FEATURE_COLUMNS_PATH):
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = json.load(f)
        print(f"Feature columns loaded: {feature_columns}")
    else:
        print(f"WARNING: Feature columns file not found at {FEATURE_COLUMNS_PATH}")


def preprocess_input_manual(data):
    """
    Manual preprocessing fallback jika preprocessor.pkl tidak tersedia.
    Menggunakan mapping manual sesuai training preprocessing.
    """
    # Numerical features (di-standardize manual — tanpa scaler, pakai z-score approx)
    # Dalam produksi, sebaiknya gunakan preprocessor.pkl
    numerical_features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    # Training data statistics (approximate from dataset)
    # Ini adalah fallback — preprocessor.pkl jauh lebih akurat
    train_means = {
        'Air temperature [K]': 300.0,
        'Process temperature [K]': 310.0,
        'Rotational speed [rpm]': 1539.0,
        'Torque [Nm]': 40.0,
        'Tool wear [min]': 108.0
    }
    train_stds = {
        'Air temperature [K]': 2.0,
        'Process temperature [K]': 1.5,
        'Rotational speed [rpm]': 179.0,
        'Torque [Nm]': 10.0,
        'Tool wear [min]': 63.0
    }
    
    # Scale numerical features
    scaled = {}
    for feat in numerical_features:
        scaled[feat] = (data[feat] - train_means[feat]) / train_stds[feat]
    
    # OneHotEncode Type (drop='first' → drop 'H')
    type_val = data.get('Type', 'L')
    type_l = 1.0 if type_val == 'L' else 0.0
    type_m = 1.0 if type_val == 'M' else 0.0
    
    # Construct feature vector sesuai urutan feature_columns
    if feature_columns:
        feature_dict = {}
        for col in feature_columns:
            if col in scaled:
                feature_dict[col] = scaled[col]
            elif col == 'Type_L':
                feature_dict[col] = type_l
            elif col == 'Type_M':
                feature_dict[col] = type_m
            else:
                feature_dict[col] = 0.0
        return pd.DataFrame([feature_dict])
    else:
        # Default column order
        return pd.DataFrame([{
            'Air temperature [K]': scaled['Air temperature [K]'],
            'Process temperature [K]': scaled['Process temperature [K]'],
            'Rotational speed [rpm]': scaled['Rotational speed [rpm]'],
            'Torque [Nm]': scaled['Torque [Nm]'],
            'Tool wear [min]': scaled['Tool wear [min]'],
            'Type_L': type_l,
            'Type_M': type_m
        }])


# ============================================================
# Routes
# ============================================================
@app.route('/', methods=['GET'])
def home():
    """Health check endpoint."""
    return jsonify({
        "service": "Predictive Maintenance API",
        "status": "running",
        "author": "Mohamad Dwi Rezky Desyafa",
        "endpoints": {
            "/": "Health check (GET)",
            "/predict": "Predict machine failure (POST)",
            "/metrics": "Prometheus metrics (GET)"
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict machine failure dari input JSON."""
    total_prediction_requests.inc()
    start_time = time.time()
    
    try:
        # Parse input
        data = request.get_json(force=True)
        
        # Validasi input
        missing = [f for f in RAW_FEATURES if f not in data]
        if missing:
            failed_prediction_requests.inc()
            return jsonify({
                "error": f"Missing features: {missing}",
                "required_features": RAW_FEATURES,
                "example": {
                    "Type": "L",
                    "Air temperature [K]": 298.1,
                    "Process temperature [K]": 308.6,
                    "Rotational speed [rpm]": 1551,
                    "Torque [Nm]": 42.8,
                    "Tool wear [min]": 0
                }
            }), 400
        
        # Cek model loaded
        if model is None:
            failed_prediction_requests.inc()
            return jsonify({"error": "Model not loaded. Check server logs."}), 503
        
        # Preprocessing
        if preprocessor is not None:
            # Gunakan preprocessor.pkl (lebih akurat)
            input_df = pd.DataFrame([{f: data[f] for f in RAW_FEATURES}])
            X_processed = preprocessor.transform(input_df)
            
            # Konversi ke DataFrame dengan nama kolom yang benar
            if feature_columns:
                X_processed = pd.DataFrame(X_processed, columns=feature_columns)
        else:
            # Fallback manual preprocessing
            X_processed = preprocess_input_manual(data)
        
        # Prediksi
        prediction = int(model.predict(X_processed)[0])
        label = "Machine Failure" if prediction == 1 else "Normal"
        
        # Update Prometheus counters
        if prediction == 1:
            predicted_machine_failure_total.inc()
        else:
            predicted_normal_machine_total.inc()
        
        successful_prediction_requests.inc()
        
        # Hitung latency
        latency = time.time() - start_time
        prediction_latency_seconds.observe(latency)
        
        return jsonify({
            "prediction": prediction,
            "label": label
        })
    
    except Exception as e:
        failed_prediction_requests.inc()
        latency = time.time() - start_time
        prediction_latency_seconds.observe(latency)
        
        return jsonify({
            "error": str(e),
            "message": "Prediction failed. Check input format."
        }), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE INFERENCE API")
    print("Nama: Mohamad Dwi Rezky Desyafa")
    print("=" * 60)
    
    # Load model dan preprocessor
    load_model()
    
    # Jalankan Flask
    port = int(os.environ.get("PORT", 5001))
    print(f"\nStarting Flask API on port {port}...")
    print(f"Endpoints:")
    print(f"  GET  http://localhost:{port}/")
    print(f"  POST http://localhost:{port}/predict")
    print(f"  GET  http://localhost:{port}/metrics")
    print(f"\nExample curl command:")
    print(f'  curl -X POST http://localhost:{port}/predict \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"Type":"L","Air temperature [K]":298.1,"Process temperature [K]":308.6,"Rotational speed [rpm]":1551,"Torque [Nm]":42.8,"Tool wear [min]":0}}\'')
    
    app.run(host='0.0.0.0', port=port, debug=False)
