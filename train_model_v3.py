import argparse
import pickle
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# ==========================================================
# KONFIGURASI
# ==========================================================

TICKERS = ["BBCA", "BBRI", "BBNI", "BMRI"]
YF_SUFFIX = ".JK"

DEFAULT_START = "2000-01-01"
DEFAULT_END = date.today().isoformat()

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42
N_CV_SPLITS = 5


# ==========================================================
# DOWNLOAD DATA
# ==========================================================

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

    df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"],
        inplace=True
    )

    df.to_csv(f"{ticker.lower()}_cleaned_pure70.csv")

    return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    c = df["Close"]

    # Return
    df["Return"] = c.pct_change()

    # SMA
    df["SMA_20"] = c.rolling(window=20).mean()
    df["SMA_50"] = c.rolling(window=50).mean()

    # EMA
    df["EMA_20"] = c.ewm(
        span=20,
        adjust=False
    ).mean()

    # RSI 14
    delta = c.diff()

    gain = delta.clip(
        lower=0
    ).rolling(window=14).mean()

    loss = (
        -delta.clip(upper=0)
    ).rolling(window=14).mean()

    rs = gain / (loss + 1e-9)

    df["RSI_14"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = c.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = c.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    # Target
    # 1 = UP
    # 0 = DOWN
    next_close = c.shift(-1)

    df["Target"] = np.where(
        next_close.notna(),
        (next_close > c).astype(int),
        np.nan
    )

    return df


# ==========================================================
# FITUR
# ==========================================================

def select_features(df: pd.DataFrame) -> list:

    return [
        "Return",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "MACD",
    ]


# ==========================================================
# MODEL
# ==========================================================

def build_model():

    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )


def calculate_roc_auc(y_true, probabilities):

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        probabilities
    )


# ==========================================================
# TRAINING
# ==========================================================

def train_ticker(ticker: str, start: str, end: str):

    print(f"\n{'=' * 64}")
    print(f"{ticker} — PURE 70/15/15")
    print(f"{'=' * 64}")

    try:
        df = download_data(
            ticker,
            start,
            end
        )

    except Exception as e:
        print(f"Error download data: {e}")
        return

    # Feature Engineering
    df = add_features(df)

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(inplace=True)

    df["Target"] = df["Target"].astype(int)

    df.to_csv(
        f"{ticker.lower()}_features_pure70.csv"
    )

    if len(df) < 200:
        print("Data terlalu sedikit, skip.")
        return

    features = select_features(df)

    X = df[features]
    y = df["Target"]

    # ======================================================
    # SPLIT 70 / 15 / 15
    # ======================================================

    train_end = int(
        len(df) * TRAIN_RATIO
    )

    validation_end = int(
        len(df) * (
            TRAIN_RATIO + VALIDATION_RATIO
        )
    )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[
        train_end:validation_end
    ]

    y_validation = y.iloc[
        train_end:validation_end
    ]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    print("\nPembagian Data:")
    print(f"Training   : {len(X_train)}")
    print(f"Validation : {len(X_validation)}")
    print(f"Testing    : {len(X_test)}")
    print(f"Total      : {len(df)}")

    # ======================================================
    # GRID SEARCH + TIME SERIES SPLIT
    # HANYA TRAINING 70%
    # ======================================================

    print(
        "\nGrid Search + Time Series Split"
    )

    tscv = TimeSeriesSplit(
        n_splits=N_CV_SPLITS
    )

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [6, 8, None],
        "min_samples_leaf": [5, 10],
        "max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=build_model(),
        param_grid=param_grid,
        cv=tscv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_params = grid_search.best_params_
    best_cv_f1 = grid_search.best_score_

    print("\nBest Parameters:")
    print(best_params)

    print(
        f"Best CV F1 Score: "
        f"{best_cv_f1:.4f}"
    )

    # ======================================================
    # MODEL PURE 70%
    # ======================================================

    model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    # HANYA TRAINING 70%
    model.fit(
        X_train,
        y_train
    )

    # ======================================================
    # VALIDATION 15%
    # ======================================================

    validation_preds = model.predict(
        X_validation
    )

    validation_probas = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    val_accuracy = accuracy_score(
        y_validation,
        validation_preds
    )

    val_precision = precision_score(
        y_validation,
        validation_preds,
        zero_division=0
    )

    val_recall = recall_score(
        y_validation,
        validation_preds,
        zero_division=0
    )

    val_f1 = f1_score(
        y_validation,
        validation_preds,
        zero_division=0
    )

    val_auc = calculate_roc_auc(
        y_validation,
        validation_probas
    )

    val_cm = confusion_matrix(
        y_validation,
        validation_preds
    )

    print("\nVALIDATION RESULTS")
    print(f"Accuracy : {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall   : {val_recall:.4f}")
    print(f"F1-Score : {val_f1:.4f}")
    print(f"ROC-AUC  : {val_auc:.4f}")

    print("\nValidation Confusion Matrix:")
    print(val_cm)

    # ======================================================
    # TESTING 15%
    # MODEL MASIH HANYA TRAINED PADA 70%
    # ======================================================

    preds = model.predict(
        X_test
    )

    probas = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        preds,
        zero_division=0
    )

    auc = calculate_roc_auc(
        y_test,
        probas
    )

    cm = confusion_matrix(
        y_test,
        preds
    )

    print("\nTEST RESULTS — PURE 70%")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nTest Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            preds,
            target_names=["Down", "Up"],
            zero_division=0
        )
    )

    # ======================================================
    # SIMPAN MODEL
    # ======================================================

    with open(
        f"model_{ticker}_pure70.pkl",
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )

    with open(
        f"features_{ticker}_pure70.pkl",
        "wb"
    ) as f:

        pickle.dump(
            features,
            f
        )

    # ======================================================
    # REPORT
    # ======================================================

    report_path = (
        f"report_{ticker}_pure70.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Training Report — {ticker}\n"
        )

        f.write(
            "VERSI 1: PURE 70/15/15\n"
        )

        f.write(
            "Model testing dilatih hanya "
            "menggunakan 70% training\n\n"
        )

        f.write(
            f"Generated: {pd.Timestamp.now()}\n"
        )

        f.write(
            f"Total data: {len(df)}\n"
        )

        f.write(
            f"Training (70%): "
            f"{len(X_train)}\n"
        )

        f.write(
            f"Validation (15%): "
            f"{len(X_validation)}\n"
        )

        f.write(
            f"Testing (15%): "
            f"{len(X_test)}\n"
        )

        f.write(
            f"\nBest Parameters:\n"
            f"{best_params}\n"
        )

        f.write(
            f"Best CV F1 Score: "
            f"{best_cv_f1:.4f}\n"
        )

        f.write(
            "\nVALIDATION RESULTS\n"
        )

        f.write(
            f"Accuracy : {val_accuracy:.4f}\n"
            f"Precision: {val_precision:.4f}\n"
            f"Recall   : {val_recall:.4f}\n"
            f"F1-Score : {val_f1:.4f}\n"
            f"ROC-AUC  : {val_auc:.4f}\n"
            f"Confusion Matrix:\n{val_cm}\n"
        )

        f.write(
            "\nTEST RESULTS\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall   : {recall:.4f}\n"
            f"F1-Score : {f1:.4f}\n"
            f"ROC-AUC  : {auc:.4f}\n"
            f"Confusion Matrix:\n{cm}\n"
        )

    print(
        f"\nReport disimpan ke {report_path}"
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        choices=TICKERS + ["ALL"],
        default="ALL"
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END
    )

    args = parser.parse_args()

    tickers = (
        TICKERS
        if args.ticker == "ALL"
        else [args.ticker]
    )

    for t in tickers:

        train_ticker(
            t,
            args.start,
            args.end
        )

    print("\nSelesai semua — PURE 70/15/15.")