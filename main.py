import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = FastAPI(title="StockMind AI Research Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

# Simple in-memory cache (10 minutes TTL per ticker)
CACHE_TTL_SECONDS = 600
_ANALYSIS_CACHE = {}


def build_features(df: pd.DataFrame):
    data = df.copy()
    data['Return_1d'] = data['Close'].pct_change()
    data['Return_5d'] = data['Close'].pct_change(5)
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['Price_vs_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    data['SMA20_vs_SMA50'] = (data['SMA_20'] - data['SMA_50']) / data['SMA_50']
    data['Volatility_10d'] = data['Return_1d'].rolling(10).std()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    future_close = data['Close'].shift(-5)
    data['Target'] = (future_close > data['Close']).astype(int)

    feature_cols = [
        'Return_1d', 'Return_5d', 'Price_vs_SMA20',
        'SMA20_vs_SMA50', 'Volatility_10d', 'RSI_14'
    ]
    return data, feature_cols


def run_ml_prediction(df: pd.DataFrame):
    data, feature_cols = build_features(df)
    latest_row = data.dropna(subset=feature_cols).iloc[-1:]
    train_df = data.dropna(subset=feature_cols + ['Target'])

    if len(train_df) < 120 or latest_row.empty:
        return None

    X = train_df[feature_cols]
    y = train_df['Target']

    split_idx = int(len(train_df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    eval_model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    eval_model.fit(X_train, y_train)
    test_acc = round(float(accuracy_score(y_test, eval_model.predict(X_test))) * 100, 1)

    full_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    full_model.fit(X, y)

    prob_up = round(float(full_model.predict_proba(latest_row[feature_cols])[0][1]) * 100, 1)
    bias = "Bullish Bias" if prob_up >= 55 else ("Bearish Bias" if prob_up <= 45 else "Sideways / Neutral")

    return {
        "horizon_days": 5,
        "up_probability_pct": prob_up,
        "ml_bias": bias,
        "holdout_accuracy_pct": test_acc
    }


def analyze_news_sentiment(stock):
    positive_words = [
        "surge", "soar", "jump", "gain", "rise", "rally", "profit", "beat",
        "strong", "growth", "upgrade", "buy", "bullish", "record", "win", "expand"
    ]
    negative_words = [
        "fall", "drop", "plunge", "slump", "loss", "miss", "weak", "downgrade",
        "sell", "bearish", "cut", "warn", "probe", "penalty", "lawsuit", "decline"
    ]

    headlines = []
    pos_hits = 0
    neg_hits = 0

    try:
        raw_news = stock.news or []
    except Exception:
        raw_news = []

    for item in raw_news[:6]:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        title = content.get("title") or (item.get("title") if isinstance(item, dict) else "")
        pub_date = content.get("pubDate", "")
        url = ""
        canonical = content.get("canonicalUrl")
        click_through = content.get("clickThroughUrl")
        if isinstance(canonical, dict) and canonical.get("url"):
            url = canonical["url"]
        elif isinstance(click_through, dict) and click_through.get("url"):
            url = click_through["url"]

        if not title:
            continue

        lower_t = title.lower()
        p_count = sum(1 for w in positive_words if w in lower_t)
        n_count = sum(1 for w in negative_words if w in lower_t)
        pos_hits += p_count
        neg_hits += n_count

        if p_count > n_count:
            tone = "Positive"
        elif n_count > p_count:
            tone = "Negative"
        else:
            tone = "Neutral"

        headlines.append({
            "title": title,
            "tone": tone,
            "pub_date": pub_date[:10] if pub_date else "",
            "url": url
        })

    net_diff = pos_hits - neg_hits
    sentiment_score = max(0, min(100, 50 + (net_diff * 10)))

    if sentiment_score >= 60:
        overall = "Positive"
    elif sentiment_score <= 40:
        overall = "Negative"
    else:
        overall = "Neutral"

    return {
        "overall_sentiment": overall,
        "sentiment_score": sentiment_score,
        "headlines": headlines
    }


def evaluate_single_ticker(ticker: str):
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return {"error": "Empty ticker"}

    now = time.time()
    cached = _ANALYSIS_CACHE.get(clean_ticker)
    if cached and (now - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]

    stock = yf.Ticker(clean_ticker)
    df = stock.history(period="2y")

    if df.empty or len(df) < 60:
        return {"ticker": clean_ticker, "error": "Invalid Ticker or Insufficient Historical Data"}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    sma_20_series = df['Close'].rolling(window=20).mean()
    sma_50_series = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))

    current_price = round(float(latest['Close']), 2)
    prev_close = round(float(prev['Close']), 2)
    change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
    rsi = round(float(rsi_series.iloc[-1]), 2)
    sma_20 = round(float(sma_20_series.iloc[-1]), 2)
    sma_50 = round(float(sma_50_series.iloc[-1]), 2)

    score = 50
    signals = []

    if current_price > sma_20:
        score += 15
        signals.append("Price above 20-Day SMA (Short-term Bullish)")
    else:
        score -= 10
        signals.append("Price below 20-Day SMA (Short-term Weak)")

    if sma_20 > sma_50:
        score += 15
        signals.append("20 SMA > 50 SMA (Uptrend Structure)")

    if rsi < 35:
        score += 15
        signals.append(f"RSI is {rsi} (Oversold Zone - Potential Bounce)")
    elif rsi > 70:
        score -= 15
        signals.append(f"RSI is {rsi} (Overbought Zone - Caution)")
    else:
        signals.append(f"RSI is {rsi} (Neutral Momentum)")

    ml_result = run_ml_prediction(df)
    news_result = analyze_news_sentiment(stock)
    if news_result["overall_sentiment"] == "Positive":
        score += 5
        signals.append(f"Recent News Sentiment is Positive ({news_result['sentiment_score']}/100)")
    elif news_result["overall_sentiment"] == "Negative":
        score -= 5
        signals.append(f"Recent News Sentiment is Negative ({news_result['sentiment_score']}/100)")

    chart_df = df.tail(90).copy()
    chart_df['SMA_20'] = sma_20_series.tail(90)
    chart_df['SMA_50'] = sma_50_series.tail(90)

    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in chart_df.index],
        "close": [round(float(x), 2) if pd.notna(x) else None for x in chart_df['Close']],
        "sma_20": [round(float(x), 2) if pd.notna(x) else None for x in chart_df['SMA_20']],
        "sma_50": [round(float(x), 2) if pd.notna(x) else None for x in chart_df['SMA_50']],
    }

    result = {
        "ticker": clean_ticker,
        "current_price": current_price,
        "change_percent": change_pct,
        "indicators": {
            "RSI_14": rsi,
            "SMA_20": sma_20,
            "SMA_50": sma_50
        },
        "research_score": max(0, min(100, score)),
        "ml_prediction": ml_result,
        "news_sentiment": news_result,
        "signals": signals,
        "chart_data": chart_data
    }

    _ANALYSIS_CACHE[clean_ticker] = {
        "timestamp": now,
        "data": result
    }
    return result


@app.get("/")
def serve_dashboard():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return JSONResponse(status_code=404, content={"error": "index.html not found"})


@app.get("/analyze/{ticker}")
def analyze_stock(ticker: str):
    return evaluate_single_ticker(ticker)


@app.get("/scan")
def scan_watchlist(tickers: str = "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS"):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = []
    for t in ticker_list[:15]:
        res = evaluate_single_ticker(t)
        results.append(res)

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    valid.sort(key=lambda x: x["research_score"], reverse=True)

    return {"results": valid + errors}
