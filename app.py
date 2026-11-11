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
        if len(hist_daily) >= 2:
            prev_day_close = float(hist_daily["Close"].iloc[-2])
        else:
            prev_day_close = float(hist_daily["Close"].iloc[-1])

        hist_weekly = tk.history(period="6mo", interval="1wk")
        if len(hist_weekly) >= 2:
            prev_week_close = float(hist_weekly["Close"].iloc[-2])
        else:
            prev_week_close = prev_day_close

        hist_monthly = tk.history(period="1y", interval="1mo")
        if len(hist_monthly) >= 2:
            prev_month_close = float(hist_monthly["Close"].iloc[-2])
        else:
            prev_month_close = prev_week_close

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

        weekly_iv = get_atm_iv_for_dte(tk.option_chain(weekly_exp), live_price)
        monthly_iv = get_atm_iv_for_dte(tk.option_chain(monthly_exp), live_price)

        if not monthly_iv:
            log_rets = np.log(hist_daily["Close"] / hist_daily["Close"].shift(1))
            monthly_iv = float(log_rets.std() * np.sqrt(252))
        if not weekly_iv:
            weekly_iv = monthly_iv

        w_dte = exp_dtes.get(weekly_exp, 7)
        m_dte = exp_dtes.get(monthly_exp, 30)

        vol_data = {}
        vol_data["Daily"] = {
            "annual_iv": weekly_iv,
            "period_iv": weekly_iv * math.sqrt(1.0 / 252.0),
            "exp": weekly_exp,
            "dte": exp_dtes.get(weekly_exp, 1),
        }
        vol_data["Weekly"] = {
            "annual_iv": weekly_iv,
            "period_iv": weekly_iv * math.sqrt(float(w_dte) / 365.0),
            "exp": weekly_exp,
            "dte": w_dte,
        }
        vol_data["Monthly"] = {
            "annual_iv": monthly_iv,
            "period_iv": monthly_iv * math.sqrt(float(m_dte) / 365.0),
            "exp": monthly_exp,
            "dte": m_dte,
        }

        return live_price, baselines, vol_data, None
    except Exception as e:
        return None, None, None, f"Data fetch error: {str(e)}"


# ---------------------------------------------------------
# Plotly Bell Curve Generator
# ---------------------------------------------------------
def build_bell_curve(anchor_close: float, period_iv: float, live_price: float, title: str):
    """Generates a normal distribution curve with ±1σ, ±2σ, ±3σ bands and live spot placement."""
    sigma = anchor_close * period_iv
    mu = anchor_close

    span = 3.6 * sigma
    x = np.linspace(mu - span, mu + span, 300)
    y = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fig = go.Figure()

    # Probability Band Shading
    fig.add_vrect(
        x0=mu - sigma,
        x1=mu + sigma,
        fillcolor="rgba(34, 197, 94, 0.16)",
        line_width=0,
        annotation_text="1σ (68.3%)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#86efac",
    )
    fig.add_vrect(
        x0=mu - 2 * sigma,
        x1=mu - sigma,
        fillcolor="rgba(234
