# Workflow CI — Predictive Maintenance

## Nama: Mohamad Dwi Rezky Desyafa

### Deskripsi

Repository ini berisi **MLflow Project** untuk training model Predictive Maintenance secara otomatis melalui **GitHub Actions CI**.

### Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── MLProject/
│   ├── MLProject               # MLflow Project configuration
│   ├── conda.yaml              # Conda environment specification
│   ├── modelling.py            # Training script
│   └── predictive_maintenance_preprocessing/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
└── README.md
```

### MLflow Project

File `MLProject/MLProject` mendefinisikan project:
- **name:** predictive-maintenance-training
- **conda_env:** conda.yaml
- **entry_points:** main → python modelling.py

### Cara Menjalankan Lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```

Atau dari root repository:

```bash
mlflow run MLProject --env-manager=local
```

### Cara Menjalankan via GitHub Actions

1. Push repository ke GitHub (pastikan public)
2. Workflow akan otomatis berjalan pada setiap `push`
3. Bisa juga trigger manual via tab **Actions** → **Run workflow**

### Artifact CI

Workflow GitHub Actions akan:
1. Checkout repository
2. Setup Python 3.12
3. Install dependencies
4. Menjalankan `mlflow run MLProject --env-manager=local`
5. Upload artifact hasil training ke GitHub Actions Artifacts

Artifact yang dihasilkan:
- `metrics.json` — Metrik evaluasi model
- `confusion_matrix.png` — Confusion matrix
- `classification_report.txt` — Laporan klasifikasi
- `model.pkl` — Model yang dilatih
