import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live IV & Expected Move", layout="wide")
st.title("📈 Expected Move Calculator (Term Structure & Prior Closes)")

# Preset quick buttons
col_presets, _ = st.columns([3, 1])
with col_presets:
    st.write("**Quick Presets:**")
    preset_cols = st.columns(6)
    presets = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMZN"]
    for i, sym in enumerate(presets):
        if preset_cols[i].button(sym):
            st.session_state["ticker_input"] = sym

default_ticker = st.session_state.get("ticker_input", "SPY")
ticker_symbol = (
    st.text_input("Enter Ticker Symbol:", value=default_ticker).strip().upper()
)


def get_atm_iv_for_dte(chain, price: float):
    """Calculates blended ATM call and put IV for a specific option chain."""
    calls = chain.calls.dropna(subset=["strike", "impliedVolatility"])
    puts = chain.puts.dropna(subset=["strike", "impliedVolatility"])

    if calls.empty or puts.empty:
        return None

    closest_call = calls.loc[(calls["strike"] - price).abs().idxmin()]
    closest_put = puts.loc[(puts["strike"] - price).abs().idxmin()]

    c_iv = float(closest_call["impliedVolatility"])
    p_iv = float(closest_put["impliedVolatility"])

    valid = [v for v in [c_iv, p_iv] if 0.05 < v < 5.0]
    return sum(valid) / len(valid) if valid else None


@st.cache_data(ttl=60)
def fetch_market_and_vol_data(symbol: str):
    tk = yf.Ticker(symbol)

    # 1. Fetch Price History & Prior Period Baselines
    hist_daily = tk.history(period="3mo")
    if hist_daily.empty:
        return None, None, None, f"No price history found for {symbol}."

    fast_info = getattr(tk, "fast_info", None)
    live_price = (
        fast_info.last_price
        if (fast_info and hasattr(fast_info, "last_price"))
        else float(hist_daily["Close"].iloc[-1])
    )

    # Daily baseline: Prior trading day close
    if len(hist_daily) >= 2:
        prev_day_close = float(hist_daily["Close"].iloc[-2])
    else:
        prev_day_close = float(hist_daily["Close"].iloc[-1])

    # Weekly baseline: Prior week close
    hist_weekly = tk.history(period="6mo", interval="1wk")
    if len(hist_weekly) >= 2:
        prev_week_close = float(hist_weekly["Close"].iloc[-2])
    else:
        prev_week_close = prev_day_close

    # Monthly baseline: Prior month close
    hist_monthly = tk.history(period="1y", interval="1mo")
    if len(hist_monthly) >= 2:
        prev_month_close = float(hist_monthly["Close"].iloc[-2])
    else:
        prev_month_close = prev_week_close

    baselines = {
        "Daily": prev_day_close,
        "Weekly": prev_week_close,
        "Monthly": prev_month_