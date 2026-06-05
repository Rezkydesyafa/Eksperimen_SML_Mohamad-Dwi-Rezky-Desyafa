"""
Automate Preprocessing Script
Nama: Mohamad Dwi Rezky Desyafa
Dataset: AI4I 2020 Predictive Maintenance Dataset

Script ini mengotomatisasi langkah preprocessing dari notebook eksperimen.
Jalankan dari root repository:
    python preprocessing/automate_Mohamad-Dwi-Rezky-Desyafa.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(filepath=None):
    """Memuat dataset dari file CSV."""
    if filepath is None:
        # Tentukan path relatif dari root repository
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "predictive_maintenance_raw", "predictive_maintenance.csv")
    
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def clean_data(df):
    """Membersihkan data dengan menghapus kolom yang tidak diperlukan."""
    print("\n--- Cleaning Data ---")
    
    # Copy dataframe agar tidak mengubah data asli
    df_clean = df.copy()
    
    # Kolom yang harus dihapus (identifier dan potensi data leakage)
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    print(f"Dropping columns: {columns_to_drop}")
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    print(f"Data shape after cleaning: {df_clean.shape}")
    print(f"Remaining columns: {list(df_clean.columns)}")
    
    return df_clean


def preprocess_data(X):
    """
    Melakukan preprocessing pada fitur:
    - OneHotEncoder untuk kolom kategorikal (Type)
    - StandardScaler untuk kolom numerik
    """
    print("\n--- Preprocessing Data ---")
    
    # Tentukan kolom numerik dan kategorikal
    numerical_features = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    categorical_features = ['Type']
    
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Buat ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    return preprocessor, numerical_features, categorical_features


def split_data(X, y, test_size=0.2, random_state=42):
    """Membagi data menjadi training dan testing set."""
    print("\n--- Splitting Data ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    print(f"y_test distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir=None):
    """Menyimpan hasil preprocessing ke file CSV."""
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "preprocessing", "predictive_maintenance_preprocessing")
    
    print(f"\n--- Saving Preprocessed Data ---")
    print(f"Output directory: {output_dir}")
    
    # Buat folder jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan file CSV
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print("Files saved successfully:")
    for f in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size} bytes)")


def main():
    """Fungsi utama untuk menjalankan seluruh pipeline preprocessing."""
    print("=" * 60)
    print("AUTOMATE PREPROCESSING")
    print("Dataset: AI4I 2020 Predictive Maintenance")
    print("Nama: Mohamad Dwi Rezky Desyafa")
    print("=" * 60)
    
    # 1. Load data
    df = load_data()
    
    # 2. Clean data
    df_clean = clean_data(df)
    
    # 3. Pisahkan fitur dan target
    print("\n--- Separating Features and Target ---")
    target_column = 'Machine failure'
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # 4. Split data terlebih dahulu (sebelum preprocessing untuk menghindari data leakage)
    X_train_raw, X_test_raw, y_train, y_test = split_data(X, y)
    
    # 5. Preprocessing (fit pada training data, transform pada keduanya)
    preprocessor, numerical_features, categorical_features = preprocess_data(X)
    
    print("\nFitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Dapatkan nama kolom setelah transformasi
    # Nama kolom numerik tetap sama (setelah scaling)
    # Nama kolom kategorikal dari OneHotEncoder
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_columns = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + cat_columns
    
    print(f"Feature names after preprocessing: {feature_names}")
    
    # Konversi ke DataFrame
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Reset index pada y
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print(f"\nPreprocessed X_train shape: {X_train_df.shape}")
    print(f"Preprocessed X_test shape: {X_test_df.shape}")
    
    # 6. Save preprocessed data
    save_preprocessed_data(X_train_df, X_test_df, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
