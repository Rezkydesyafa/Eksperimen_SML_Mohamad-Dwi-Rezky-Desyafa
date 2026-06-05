# Membangun Model — Predictive Maintenance

## Nama: Mohamad Dwi Rezky Desyafa

### Deskripsi

Folder ini berisi script untuk membangun model machine learning menggunakan **MLflow Tracking** untuk mencatat eksperimen, parameter, metrik, dan artifact.

### Struktur Folder

```
Membangun_model/
├── modelling.py                          # Model baseline dengan autolog
├── modelling_tuning.py                   # Model dengan tuning + manual logging
├── predictive_maintenance_preprocessing/ # Dataset hasil preprocessing
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── requirements.txt                      # Dependencies
├── screenshoot_dashboard.jpg             # Screenshot MLflow dashboard
├── screenshoot_artifak.jpg               # Screenshot MLflow artifacts
└── README.md
```

### Cara Install Dependencies

```bash
pip install -r requirements.txt
```

### Cara Menjalankan MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Buka browser: http://127.0.0.1:5000

### Cara Menjalankan modelling.py (Baseline + Autolog)

```bash
python modelling.py
```

Script ini:
- Menggunakan `mlflow.sklearn.autolog()` (otomatis mencatat semua)
- Melatih `RandomForestClassifier(random_state=42, class_weight="balanced")`
- Menampilkan metrik evaluasi ke terminal

### Cara Menjalankan modelling_tuning.py (Tuning + Manual Logging)

```bash
python modelling_tuning.py
```

Script ini:
- Menggunakan `GridSearchCV` untuk hyperparameter tuning
- **Tidak** menggunakan autolog
- Manual logging: `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`, `mlflow.sklearn.log_model()`
- Menghasilkan artifact: confusion_matrix.png, classification_report.txt, feature_importance.png, best_model.pkl

### Artifact yang Dihasilkan

| Artifact | Deskripsi |
|----------|-----------|
| `confusion_matrix.png` | Visualisasi confusion matrix |
| `classification_report.txt` | Laporan klasifikasi detail |
| `feature_importance.png` | Visualisasi feature importance |
| `best_model.pkl` | Model terbaik hasil tuning |
| `feature_columns.json` | Nama kolom fitur |

### Screenshot yang Harus Diambil

1. **screenshoot_dashboard.jpg**: Buka MLflow UI → halaman daftar run → screenshot
2. **screenshoot_artifak.jpg**: Klik run → tab Artifacts → screenshot halaman artifact/model
