import datetime
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Expected Move & Volatility Desk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #94a3b8 !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Controls & Auto-Refresh
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ App Settings")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    refresh_rate = st.selectbox(
        "Refresh Interval", [30, 60, 120], index=0, format_func=lambda x: f"{x}s"
    )
    force_refresh = st.button("⚡ Force Live Refresh")

if auto_refresh:
    st.markdown(
        f"""
        <script>
            setTimeout(function(){{
               window.location.reload();
            }}, {refresh_rate * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )

if force_refresh:
    st.cache_data.clear()

# ---------------------------------------------------------
# Ticker Inputs & Presets
# ---------------------------------------------------------
st.title("📊 Volatility & Expected Move Desk")

col_presets, _ = st.columns([4, 1])
with col_presets:
    preset_cols = st.columns(6)
    presets = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMZN"]
    for i, sym in enumerate(presets):
        if preset_cols[i].button(sym):
            st.session_state["ticker_input"] = sym

default_ticker = st.session_state.get("ticker_input", "SPY")
ticker_symbol = st.text_input("Enter Ticker Symbol:", value=default_ticker).strip().upper()


# ---------------------------------------------------------
# Options Chain & Volatility Logic
# ---------------------------------------------------------
def get_atm_iv_for_dte(chain, price: float):
    """Calculates blended ATM call and put IV for a specific option chain."""
    try:
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
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_market_and_vol_data(symbol: str, bypass_cache: bool = False):
    try:
        tk = yf.Ticker(symbol)

        # 1. Price History & Baselines
        hist_daily = tk.history(period="3mo")
        if hist_daily.empty:
            return None, None, None, f"No price history found for {symbol}."

        fast_info = getattr(tk, "fast_info", None)
        if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price is not None:
            live_price = float(fast_info.last_price)
        else:
            live_price = float(hist_daily["Close"].iloc[-1])

        # Baseline anchor closes
        prev_day_close = (
            float(hist_daily["Close"].iloc[-2]) if len(hist_daily) >= 2 else float(hist_daily["Close"].iloc[-1])
        )

        hist_weekly = tk.history(period="6mo", interval="1wk")
        prev_week_close = (
            float(hist_weekly["Close"].iloc[-2]) if len(hist_weekly) >= 2 else prev_day_close
        )

        hist_monthly = tk.history(period="1y", interval="1mo")
        prev_month_close = (
            float(hist_monthly["Close"].iloc[-2]) if len(hist_monthly) >= 2 else prev_week_close
        )

        baselines = {
            "Daily": prev_day_close,
            "Weekly": prev_week_close,
            "Monthly": prev_month_close,
        }

        # 2. Term Structure Expirations
        expirations = tk.options
        if not expirations:
            return None, None, None, f"No options chains available for {symbol}."

        today = datetime.date.today()
        exp_dtes = {}
        for exp_str in expirations:
            exp_d = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_d - today).days
            if dte > 0:
                exp_dtes[exp_str] = dte

        if not exp_dtes:
            return None, None, None, "No active future option expirations found."

        weekly_exp = min(exp_dtes.keys(), key=lambda x: abs(exp_dtes[x] - 7))
        monthly_exp = min(exp_dtes.keys(), key=lambda x: abs(exp_dtes[x] - 30))

        weekly