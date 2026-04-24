"""
automate_Nama-siswa.py
======================
Script otomatisasi preprocessing dataset Heart Disease UCI.
Mengembalikan data yang sudah siap dilatih (train & test CSV).

Cara penggunaan:
    python automate_Nama-siswa.py

Output:
    heart_disease_preprocessing/heart_train_preprocessed.csv
    heart_disease_preprocessing/heart_test_preprocessed.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# KONSTANTA
# ============================================================
RAW_DATA_PATH = "heart_disease_raw/heart_cleveland_upload.csv"
OUTPUT_DIR    = "preprocessing/heart_disease_preprocessing"
TRAIN_OUTPUT  = os.path.join(OUTPUT_DIR, "heart_train_preprocessed.csv")
TEST_OUTPUT   = os.path.join(OUTPUT_DIR, "heart_test_preprocessed.csv")

NUMERICAL_COLS    = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS  = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
OHE_COLS          = ["cp", "restecg", "slope", "thal"]
TARGET_COL        = "condition"
TEST_SIZE         = 0.2
RANDOM_STATE      = 42


# ============================================================
# FUNGSI-FUNGSI PREPROCESSING
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    print(f"[1/6] 📂 Memuat dataset dari {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"      Shape awal: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi missing values: median untuk numerik, modus untuk kategorikal."""
    print("[2/6] 🔧 Menangani missing values ...")
    df = df.copy()
    total_missing = df.isnull().sum().sum()

    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"      → {col}: diisi median ({median_val:.2f})")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"      → {col}: diisi modus ({mode_val})")

    if total_missing == 0:
        print("      ✅ Tidak ada missing values ditemukan.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    print("[3/6] 🗑️  Menghapus duplikat ...")
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"      {n_dup} duplikat dihapus. Shape sekarang: {df.shape}")
    return df


def handle_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Meng-clip outlier pada fitur numerik menggunakan batas IQR."""
    print("[4/6] 📊 Menangani outlier (IQR) ...")
    df = df.copy()
    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"      → {col}: {n_out} outlier di-clip ke [{lower:.2f}, {upper:.2f}]")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot Encoding untuk fitur kategorikal multi-kelas."""
    print("[5/6] 🔠 Encoding fitur kategorikal ...")
    cols_to_encode = [c for c in OHE_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)
    print(f"      Kolom setelah encoding: {len(df.columns)} kolom")
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Memisahkan fitur & target, train-test split, lalu standarisasi.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    scaler : StandardScaler yang sudah di-fit
    """
    print("[6/6] ✂️  Split & Standarisasi ...")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    num_available = [c for c in NUMERICAL_COLS if c in X_train.columns]
    X_train[num_available] = scaler.fit_transform(X_train[num_available])
    X_test[num_available]  = scaler.transform(X_test[num_available])

    print(f"      X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


def save_results(X_train, X_test, y_train, y_test):
    """Menyimpan dataset yang sudah diproses ke CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    train_df.to_csv(TRAIN_OUTPUT, index=False)

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_OUTPUT, index=False)

    print("\n✅ Dataset berhasil disimpan:")
    print(f"   Train → {TRAIN_OUTPUT}  ({train_df.shape})")
    print(f"   Test  → {TEST_OUTPUT}  ({test_df.shape})")


# ============================================================
# PIPELINE UTAMA
# ============================================================

def preprocess_pipeline(raw_path: str = RAW_DATA_PATH):
    """Menjalankan seluruh tahapan preprocessing secara berurutan."""
    print("=" * 55)
    print("  OTOMATISASI PREPROCESSING - Heart Disease UCI")
    print("=" * 55)

    df = load_data(raw_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers_iqr(df)
    df = encode_categorical(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    save_results(X_train, X_test, y_train, y_test)

    print("\n🎉 Preprocessing selesai! Data siap untuk pelatihan model.")
    print("=" * 55)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    preprocess_pipeline()
