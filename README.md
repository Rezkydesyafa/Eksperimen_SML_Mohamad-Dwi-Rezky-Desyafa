# Eksperimen SML — Mohamad Dwi Rezky Desyafa

## Dataset: Predictive Maintenance Dataset AI4I 2020

### Deskripsi

Repository ini berisi eksperimen dataset dan preprocessing untuk project **Predictive Maintenance Machine Failure**. Dataset yang digunakan adalah **AI4I 2020 Predictive Maintenance Dataset** dari Kaggle.

**Jenis Task:** Binary Classification  
**Target:** `Machine failure` (0=Normal, 1=Failure)

### Struktur Folder

```
Eksperimen_SML_Mohamad-Dwi-Rezky-Desyafa/
├── predictive_maintenance_raw/
│   └── predictive_maintenance.csv
├── preprocessing/
│   ├── Eksperimen_Mohamad-Dwi-Rezky-Desyafa.ipynb
│   ├── automate_Mohamad-Dwi-Rezky-Desyafa.py
│   └── predictive_maintenance_preprocessing/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
└── README.md
```

### Cara Menjalankan Notebook

1. Buka notebook di Jupyter atau Google Colab:
   ```
   preprocessing/Eksperimen_Mohamad-Dwi-Rezky-Desyafa.ipynb
   ```
2. Jalankan semua cell secara berurutan.
3. Hasil preprocessing akan disimpan di `preprocessing/predictive_maintenance_preprocessing/`.

### Cara Menjalankan Script Otomatisasi

Jalankan dari root repository:

```bash
python preprocessing/automate_Mohamad-Dwi-Rezky-Desyafa.py
```

### Output Preprocessing

| File | Deskripsi |
|------|-----------|
| `X_train.csv` | Fitur training (sudah di-scale dan di-encode) |
| `X_test.csv` | Fitur testing (sudah di-scale dan di-encode) |
| `y_train.csv` | Target training |
| `y_test.csv` | Target testing |

### Fitur Setelah Preprocessing

| Fitur | Transformasi |
|-------|-------------|
| Air temperature [K] | StandardScaler |
| Process temperature [K] | StandardScaler |
| Rotational speed [rpm] | StandardScaler |
| Torque [Nm] | StandardScaler |
| Tool wear [min] | StandardScaler |
| Type_L | OneHotEncoder (drop='first') |
| Type_M | OneHotEncoder (drop='first') |
