"""
app.py — Flask Backend untuk Analisis Prediksi Saham BIG4
=========================================================

Fungsi utama:
1. Memuat model Random Forest yang telah dilatih.
2. Mengambil data historis saham dari Yahoo Finance.
3. Menghitung indikator teknikal berdasarkan data historis dan input pengguna.
4. Melakukan prediksi arah pergerakan harga saham.
5. Mengirim hasil prediksi dalam format JSON ke frontend.

Support:
- BBCA
- BBRI
- BBNI
- BMRI
"""

# =========================
# Import Library
# =========================

# Flask digunakan untuk membuat backend web
from flask import Flask, render_template, request, jsonify
# Mengizinkan frontend dan backend berkomunikasi walaupun berbeda port
from flask_cors import CORS
# Digunakan untuk mengolah data
import pandas as pd
import numpy as np
# Digunakan untuk membaca file model (.pkl)
import pickle
# Digunakan untuk pengecekan file
import os
# Mengambil data saham dari Yahoo Finance
import yfinance as yf
# Mengolah tanggal
from datetime import datetime, timedelta

# =========================
# Inisialisasi Flask
# =========================

# Membuat objek aplikasi Flask
app = Flask(__name__)

# Mengaktifkan CORS agar frontend dapat mengakses backend
CORS(app, resources={r"/*": {"origins": "*"}})


# =========================
# Load Model Random Forest
# =========================

# Daftar saham yang didukung sistem
TICKERS = ["BBCA", "BBRI", "BBNI", "BMRI"]
# Dictionary untuk menyimpan model beserta fitur yang digunakan
MODELS  = {}

# Melakukan loading seluruh model ketika aplikasi dijalankan
for ticker in TICKERS:
    try:
        # Membaca model Random Forest
        # serta daftar fitur yang digunakan model
        MODELS[ticker] = {
            "model":    pickle.load(open(f"model_{ticker}.pkl",    "rb")),
            "features": pickle.load(open(f"features_{ticker}.pkl", "rb")),
        }
        print(f"  ✅ Model {ticker} loaded")
    except FileNotFoundError:
        # Jika model belum tersedia
        print(f"  ⚠️  Model {ticker} belum ada — jalankan train_model_v3.py dulu")
    except Exception as e:
        # Jika terjadi kesalahan ketika membaca model
        print(f"  ❌ Error loading {ticker}: {e}")

print(f"\n  Model siap: {list(MODELS.keys())}\n")

# ==========================================================
# Fungsi Perhitungan Indikator Teknikal
# ==========================================================

def calculate_features(ticker, input_date, open_price, high_price, low_price, close_price):
    """
    Fungsi ini bertugas:

    1. Mengambil data historis saham dari Yahoo Finance.
    2. Menambahkan data OHLC yang dimasukkan pengguna.
    3. Menghitung indikator teknikal.
    4. Mengembalikan data fitur yang akan digunakan model.
    """
    # Mengambil data historis sekitar 120 hari
    start = (pd.to_datetime(input_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(input_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Download data saham
    df = yf.download(
        ticker + ".JK",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # Jika format kolom MultiIndex maka diubah menjadi satu level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Jika data kosong maka tampilkan error
    if df.empty:
        raise Exception("Data Yahoo Finance tidak ditemukan")

    input_date = pd.to_datetime(input_date)

    # ==================================================
    # Menambahkan data input pengguna
    # ==================================================

    # Jika tanggal belum ada pada data historis    
    if input_date not in df.index:

        new_row = pd.DataFrame({
            "Open":[open_price],
            "High":[high_price],
            "Low":[low_price],
            "Close":[close_price],
            "Volume":[0]
        }, index=[input_date])

        df = pd.concat([df,new_row])

    else:
        # Jika tanggal sudah ada maka update nilainya
        df.loc[input_date,"Open"] = open_price
        df.loc[input_date,"High"] = high_price
        df.loc[input_date,"Low"] = low_price
        df.loc[input_date,"Close"] = close_price

    # Mengurutkan data berdasarkan tanggal
    df = df.sort_index()

    # Mengambil harga penutupan
    c = df["Close"]

    # ==================================================
    # Menghitung RSI
    # ==================================================

    delta = c.diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain/(loss+1e-9)

    df["RSI_14"] = 100-(100/(1+rs))

    # ==================================================
    # Menghitung MACD
    # ==================================================

    ema12 = c.ewm(span=12,adjust=False).mean()

    ema26 = c.ewm(span=26,adjust=False).mean()

    df["MACD"] = ema12-ema26

    # ==================================================
    # Menghitung Return
    # ==================================================

    df["Return"] = c.pct_change()

    # ==================================================
    # Menghitung Moving Average
    # ==================================================

    df["SMA_20"] = c.rolling(20).mean()

    df["SMA_50"] = c.rolling(50).mean()

    df["EMA_20"] = c.ewm(span=20,adjust=False).mean()

    # Mengambil hanya baris sesuai tanggal input

    row = df.loc[[input_date]]

    return row

# ==========================================================
# Halaman Utama
# ==========================================================

@app.route("/")
def home():
    """
    Menampilkan halaman utama website.
    """
    return render_template("index.html")

# ==========================================================
# Endpoint Status Model
# ==========================================================


@app.route("/status")
def status():
    """
    Mengecek model mana saja yang berhasil dimuat
    ketika backend dijalankan.
    """

    return jsonify({
        ticker: (ticker in MODELS)
        for ticker in TICKERS
    })

# ==========================================================
# Endpoint Prediksi
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama yang melakukan proses prediksi.

    Tahapan proses:
    1. Membaca input pengguna.
    2. Validasi data.
    3. Menghitung indikator teknikal.
    4. Memilih fitur sesuai model.
    5. Melakukan prediksi Random Forest.
    6. Mengirim hasil ke frontend.
    """
    try:
        # ===========================
        # Membaca input dari form
        # ===========================
        ticker = request.form.get("ticker", "BBCA").upper().strip()
        input_date = request.form.get("date")
        o = float(request.form.get("open",  0))
        h = float(request.form.get("high",  0))
        l = float(request.form.get("low",   0))
        c = float(request.form.get("close", 0))

        # ===========================
        # Validasi Data
        # ===========================

        # Memastikan ticker tersedia
        if ticker not in TICKERS:
            return jsonify({"error": f"Ticker tidak dikenal: {ticker}"}), 400

        # Memastikan model tersedia
        if ticker not in MODELS:
            return jsonify({
                "error": f"Model {ticker} belum ditraining. "
                         f"Jalankan: python train_model_v3.py --ticker {ticker}"
            }), 503

        # Memastikan harga valid
        if any(v <= 0 for v in [o, h, l, c]):
            return jsonify({"error": "Semua harga harus > 0"}), 400

        if h < l:
            return jsonify({"error": "High harus >= Low"}), 400

        model    = MODELS[ticker]["model"]
        features = MODELS[ticker]["features"]
    
        # ===========================
        # Menghitung indikator teknikal
        # ===========================
        
        feature_row = calculate_features(
            ticker=ticker,
            input_date=input_date,
            open_price=o,
            high_price=h,
            low_price=l,
            close_price=c
        )
        # Mengambil fitur yang memang digunakan model
        row_s = feature_row[features]
        
        # ===========================
        # Prediksi Random Forest
        # ===========================
        pred        = int(model.predict(row_s)[0])
        proba       = model.predict_proba(row_s)[0]

        # Mengubah probabilitas menjadi persen
        prob_up     = float(proba[1]) * 100
        prob_down   = float(proba[0]) * 100
        confidence  = round(max(prob_up, prob_down), 2)
        # Menentukan arah prediksi
        direction   = "UP" if pred == 1 else "DOWN"

        # ===========================
        # Interpretasi indikator
        # ===========================

        # Mengambil nilai indikator teknikal       
        rsi = float(feature_row["RSI_14"].iloc[0])
        macd = float(feature_row["MACD"].iloc[0])
        ret = float(feature_row["Return"].iloc[0])

        # Menentukan tingkat volatilitas
        vol = abs((h-l)/c)
        if vol < 0.015:
            vol_lv = "Rendah"
        elif vol < 0.03:
            vol_lv = "Sedang"
        else:
            vol_lv = "Tinggi"
        
        # Menentukan kondisi RSI
        if rsi > 70:
            mom_lv = "Overbought"
        elif rsi < 30:
            mom_lv = "Oversold"
        else:
            mom_lv = "Netral"
        
        # Menentukan sentimen prediksi
        sen_lv = "Positif" if pred==1 else "Negatif"
        
        analysis = (
            f"RSI = {rsi:.2f}, "
            f"MACD = {macd:.2f}, "
            f"Return = {ret*100:.2f}%. "
            f"Model memprediksi probabilitas UP {prob_up:.2f}% "
            f"dan DOWN {prob_down:.2f}%."
        )

        # ===========================
        # Mengirim hasil ke frontend
        # ===========================
        return jsonify({
            "direction":  direction,
            "confidence": confidence,
            "proba_up":   round(prob_up,   2),
            "proba_down": round(prob_down, 2),
            "technical":{
                 "RSI":round(rsi,2),
                 "MACD":round(macd,2),
                 "Return":round(ret*100,2),
            },
            "analysis":   analysis,
            "metrics": {
                "volatilitas": vol_lv,
                "momentum":    mom_lv,
                "sentimen":    sen_lv,
            }
        })

    except ValueError as e:
        return jsonify({"error": f"Input tidak valid: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ─── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = True,
    )