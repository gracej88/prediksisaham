"""
app.py — Flask Backend untuk Analisis Saham BIG4
=================================================
Fix:
  - CORS enabled (untuk frontend di port berbeda, misal Live Server :5500)
  - /predict return jsonify() bukan render_template
  - /status endpoint untuk cek model mana yang sudah siap
  - Support BBCA, BBRI, BBNI, BMRI
  - Fallback graceful kalau model belum ada

Install:
    pip install flask flask-cors scikit-learn pandas numpy pickle5
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

# ─── CORS: izinkan request dari semua origin (termasuk :5500) ───────
# Untuk production, ganti "*" dengan domain spesifik
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── Load semua model saat startup ──────────────────────────────────
TICKERS = ["BBCA", "BBRI", "BBNI", "BMRI"]
MODELS  = {}

for ticker in TICKERS:
    try:
        MODELS[ticker] = {
            "model":    pickle.load(open(f"model_{ticker}.pkl",    "rb")),
            "features": pickle.load(open(f"features_{ticker}.pkl", "rb")),
        }
        print(f"  ✅ Model {ticker} loaded")
    except FileNotFoundError:
        print(f"  ⚠️  Model {ticker} belum ada — jalankan train_model_v3.py dulu")
    except Exception as e:
        print(f"  ❌ Error loading {ticker}: {e}")

print(f"\n  Model siap: {list(MODELS.keys())}\n")

def calculate_features(ticker, input_date, open_price, high_price, low_price, close_price):
    """
    Mengambil data historis lalu menghitung indikator teknikal
    menggunakan data aktual + data yang diinput user.
    """

    start = (pd.to_datetime(input_date) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(input_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        ticker + ".JK",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise Exception("Data Yahoo Finance tidak ditemukan")

    input_date = pd.to_datetime(input_date)

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

        df.loc[input_date,"Open"] = open_price
        df.loc[input_date,"High"] = high_price
        df.loc[input_date,"Low"] = low_price
        df.loc[input_date,"Close"] = close_price

    df = df.sort_index()

    c = df["Close"]

    delta = c.diff()

    gain = delta.clip(lower=0).rolling(14).mean()

    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain/(loss+1e-9)

    df["RSI_14"] = 100-(100/(1+rs))

    ema12 = c.ewm(span=12,adjust=False).mean()

    ema26 = c.ewm(span=26,adjust=False).mean()

    df["MACD"] = ema12-ema26

    df["Return"] = c.pct_change()

    df["SMA_20"] = c.rolling(20).mean()

    df["SMA_50"] = c.rolling(50).mean()

    df["EMA_20"] = c.ewm(span=20,adjust=False).mean()

    row = df.loc[[input_date]]

    return row

# ─── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve HTML template utama."""
    return render_template("index.html")


@app.route("/status")
def status():
    """
    Cek status model yang sudah loaded.
    Frontend bisa polling endpoint ini untuk update indikator status.
    """
    return jsonify({
        ticker: (ticker in MODELS)
        for ticker in TICKERS
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Terima OHLC + ticker, kembalikan prediksi JSON.

    Expects form data:
        ticker  : BBCA / BBRI / BBNI / BMRI
        open    : float
        high    : float
        low     : float
        close   : float

    Returns JSON:
        direction   : "UP" | "DOWN"
        confidence  : float (contoh: 67.83)
        analysis    : string
        metrics     : { volatilitas, momentum, sentimen }
        proba_up    : float  (probabilitas UP, 0–100)
        proba_down  : float  (probabilitas DOWN, 0–100)
    """
    try:
        # ── Parse input ──────────────────────────────────
        ticker = request.form.get("ticker", "BBCA").upper().strip()
        input_date = request.form.get("date")
        o = float(request.form.get("open",  0))
        h = float(request.form.get("high",  0))
        l = float(request.form.get("low",   0))
        c = float(request.form.get("close", 0))

        # ── Validasi ─────────────────────────────────────
        if ticker not in TICKERS:
            return jsonify({"error": f"Ticker tidak dikenal: {ticker}"}), 400

        if ticker not in MODELS:
            return jsonify({
                "error": f"Model {ticker} belum ditraining. "
                         f"Jalankan: python train_model_v3.py --ticker {ticker}"
            }), 503

        if any(v <= 0 for v in [o, h, l, c]):
            return jsonify({"error": "Semua harga harus > 0"}), 400

        if h < l:
            return jsonify({"error": "High harus >= Low"}), 400

        # ── Ambil artifacts ──────────────────────────────
        model    = MODELS[ticker]["model"]
        features = MODELS[ticker]["features"]
        
        feature_row = calculate_features(
            ticker=ticker,
            input_date=input_date,
            open_price=o,
            high_price=h,
            low_price=l,
            close_price=c
        )
        row_s = feature_row[features]
        
        # ── Predict ──────────────────────────────────────
        pred        = int(model.predict(row_s)[0])
        proba       = model.predict_proba(row_s)[0]   # [prob_down, prob_up]
        prob_up     = float(proba[1]) * 100
        prob_down   = float(proba[0]) * 100
        confidence  = round(max(prob_up, prob_down), 2)
        direction   = "UP" if pred == 1 else "DOWN"

        # ── Interpretasi ─────────────────────────────────
        rsi = float(feature_row["RSI_14"].iloc[0])
        macd = float(feature_row["MACD"].iloc[0])
        ret = float(feature_row["Return"].iloc[0])
        vol = abs((h-l)/c)
        if vol < 0.015:
            vol_lv = "Rendah"
        elif vol < 0.03:
            vol_lv = "Sedang"
        else:
            vol_lv = "Tinggi"
        
        if rsi > 70:
            mom_lv = "Overbought"
        elif rsi < 30:
            mom_lv = "Oversold"
        else:
            mom_lv = "Netral"
        
        sen_lv = "Positif" if pred==1 else "Negatif"
        
        analysis = (
            f"RSI = {rsi:.2f}, "
            f"MACD = {macd:.2f}, "
            f"Return = {ret*100:.2f}%. "
            f"Model memprediksi probabilitas UP {prob_up:.2f}% "
            f"dan DOWN {prob_down:.2f}%."
        )

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