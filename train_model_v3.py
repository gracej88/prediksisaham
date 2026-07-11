# Import library yang digunakan untuk pengolahan data, machine learning, dan evaluasi model
import argparse
import pickle
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

# Konfigurasi penelitian seperti daftar saham, periode data, rasio training-testing, dan parameter model
TICKERS = ["BBCA", "BBRI", "BBNI", "BMRI"]
YF_SUFFIX = ".JK"
DEFAULT_START = "2000-01-01"
DEFAULT_END = date.today().isoformat()
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
N_CV_SPLITS = 5

# Mengambil data historis saham dari Yahoo Finance berdasarkan kode saham dan periode waktu
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    yf_ticker = ticker + YF_SUFFIX
    print(f"Downloading {yf_ticker} [{start} → {end}]")
    df = yf.download(
        yf_ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"Tidak ada data untuk {yf_ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Kolom hilang: {missing}")
    df.index = pd.to_datetime(df.index)
# Membersihkan data saham dari nilai kosong sebelum digunakan pada proses analisis
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
# Menyimpan data saham yang telah dibersihkan dalam format CSV
    df.to_csv(f"{ticker.lower()}_cleaned.csv")
    print(f"Data disimpan ke {ticker.lower()}_cleaned.csv")
    return df

# Membuat fitur indikator teknikal dan menentukan target prediksi arah pergerakan saham
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
 # Menghitung indikator RSI untuk melihat kondisi momentum harga saham
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
 # Menghitung indikator MACD berdasarkan perbedaan EMA periode pendek dan panjang
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
# Membuat fitur utama berupa return, SMA, dan EMA sebagai input model
    df["Return"] = c.pct_change()
    df["SMA_20"] = c.rolling(20).mean()
    df["SMA_50"] = c.rolling(50).mean()
    df["EMA_20"] = c.ewm(span=20, adjust=False).mean()
# Menentukan target prediksi berdasarkan perbandingan harga penutupan hari berikutnya
# Nilai 1 menunjukkan harga naik dan 0 menunjukkan harga turun
    df["Target"] = (c.shift(-1) > c).astype(int)
    return df

# Menentukan indikator teknikal yang digunakan sebagai fitur input Random Forest
def select_features(df: pd.DataFrame) -> list:
    return [
        "Return",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "MACD",
    ]

# Membuat model Random Forest dengan parameter yang telah ditentukan
def build_model():
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return rf

# Melakukan proses preprocessing, training, validasi, pengujian, dan penyimpanan model
def train_ticker(ticker: str, start: str, end: str):
    print(f"\n{'='*64}\n{ticker} — Training\n{'='*64}")
    try:
        df = download_data(ticker, start, end)
    except Exception as e:
        print(f"Error download data: {e}")
        return
# Membuat indikator teknikal dan target prediksi
    df = add_features(df)
# Membersihkan data dari nilai kosong dan nilai tidak valid
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
# Mengecek jumlah data minimal agar model dapat dilatih
    if len(df) < 200:
        print("Data terlalu sedikit, skip.")
        return

# Menentukan fitur input dan target yang digunakan dalam model
    features = select_features(df)
    X = df[features]
    y = df["Target"]
    print(f"Fitur: {features}")
    print(f"Class balance: {y.mean():.2%}")

# Membagi data berdasarkan urutan waktu menjadi 80% training dan 20% testing
# Data testing tidak digunakan selama proses validasi agar evaluasi tetap objektif
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train / Test: {len(X_train)} / {len(X_test)}")

# Melakukan validasi model menggunakan Time Series Cross Validation
# Validasi hanya dilakukan pada data training untuk menjaga urutan waktu
    print("Walk-forward CV (hanya pada training set):")
    results = {"accuracy": [], "auc": []}
# Membagi data training menjadi 5 fold untuk melihat kestabilan performa model
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
# Membagi data training menjadi data latih dan validasi pada setiap fold
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
# Random Forest tidak membutuhkan proses normalisasi sehingga menggunakan data asli
        X_tr_s = X_tr
        X_val_s = X_val
# Membuat dan melatih model pada setiap fold validasi
        model = build_model()
        model.fit(X_tr_s, y_tr)
# Melakukan prediksi pada data validasi
        preds = model.predict(X_val_s)
        probas = model.predict_proba(X_val_s)[:, 1]
# Menghitung nilai evaluasi pada setiap fold
        results["accuracy"].append(accuracy_score(y_val, preds))
        results["auc"].append(roc_auc_score(y_val, probas))
        print(f"Fold {fold}: Acc={results['accuracy'][-1]:.3f} AUC={results['auc'][-1]:.3f}")

# Menampilkan rata-rata hasil validasi dari seluruh fold
    print("CV Rata-rata (training set):")
    for k, v in results.items():
        print(f"{k}: {np.mean(v):.3f} ± {np.std(v):.3f}")

# Melatih model akhir menggunakan seluruh data training (80%)
    X_train_s = X_train
    X_test_s = X_test
    model = build_model()
    model.fit(X_train_s, y_train)

# Menguji model menggunakan data testing yang belum pernah dilihat model
    preds = model.predict(X_test_s)
    probas = model.predict_proba(X_test_s)[:, 1]
# Menghitung metrik evaluasi akhir model
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)

    print(f"\nHold-out Test ({len(X_test)} samples):")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix (Down/Up):")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

 # Menyimpan model Random Forest agar dapat digunakan kembali pada dashboard
    with open(f"model_{ticker}.pkl", "wb") as f:
        pickle.dump(model, f)
# Menyimpan daftar fitur yang digunakan model
    with open(f"features_{ticker}.pkl", "wb") as f:
        pickle.dump(features, f)

# Menyimpan hasil evaluasi model dalam file laporan
    report_path = f"report_{ticker}.txt"
    with open(report_path, "w") as f:
        f.write(f"Training Report — {ticker}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Data sampai: {end}\n")
        f.write(f"Total data: {len(df)}\n")
        f.write(f"Fitur: {features}\n")
        f.write(f"Train / Test: {split_idx} / {len(X) - split_idx}\n")
        f.write(f"\nCV Results (training set only):\n")
        for k, v in results.items():
            f.write(f"{k}: {np.mean(v):.3f} ± {np.std(v):.3f}\n")
        f.write(f"\nHold-out Accuracy: {acc:.4f}\n")
        f.write(f"Hold-out ROC-AUC: {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print(f"Report disimpan ke {report_path}")

# Menjalankan program utama untuk proses training model saham
if __name__ == "__main__":
# Membuat konfigurasi input dari command line saat menjalankan program
    parser = argparse.ArgumentParser()
# Menentukan pilihan saham yang akan diproses
    parser.add_argument("--ticker", choices=TICKERS + ["ALL"], default="ALL")
# Menentukan periode awal data saham
    parser.add_argument("--start", default=DEFAULT_START)
# Menentukan periode akhir data saham
    parser.add_argument("--end", default=DEFAULT_END)
# Opsi untuk tidak melakukan download data baru
    parser.add_argument("--no-download", action="store_true")
# Mengambil nilai input dari command line
    args = parser.parse_args()

# Menentukan saham yang akan dilakukan proses training
    tickers = TICKERS if args.ticker == "ALL" else [args.ticker]
# Melakukan training model untuk setiap saham yang dipilih
    for t in tickers:
        train_ticker(t, args.start, args.end)
# Menampilkan pesan bahwa seluruh proses training telah selesai
    print("Selesai semua.")