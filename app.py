import datetime
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Expected Move & Volatility Desk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly, high-contrast trading terminal styling
st.markdown(
    """
    <style>
    /* Global font size and container padding for mobile */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
    }
    /* Button enhancements */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    /* Tab headers */
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

# JavaScript timer for automated market-hours refresh
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

# ---------------------------------------------------------
# Header & Ticker Input
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
# Option & Volatility Helper Functions
# ---------------------------------------------------------
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
def fetch_market_and_vol_data(symbol: str, bypass_cache: bool = False):
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

    # Prior Closes
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

    # 2. Match Term Structure Expirations
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

    vol_data = {
        "Daily": {
            "annual_iv": weekly_iv,
            "period_iv": weekly_iv * math.sqrt(1 / 252),
            "exp": weekly_exp,
            "dte": exp_dtes.get(weekly_exp, 1),
        },
        "Weekly": {
            "annual_iv": weekly_iv,
            "period_iv": weekly_iv * math.sqrt(exp_dtes.get(weekly_exp, 7) / 365),
            "exp": weekly_exp,
            "dte": exp_dtes.get(weekly_exp, 7),
        },
        "Monthly": {
            "annual_iv": monthly_iv,
            "period_iv": monthly_iv * math.sqrt(exp_dtes.get(monthly_exp, 30) / 365),
            "exp": monthly_exp,
            "dte": exp_dtes.get(monthly_exp, 30),
        },
    }

    return live_price, baselines, vol_data, None


# Clear cache if forced
if force_refresh:
    st.cache_data.clear()

# ---------------------------------------------------------
# Execution & Layout Rendering
# ---------------------------------------------------------
if ticker_symbol:
    with st.spinner(f"Fetching real-time market data for {ticker_symbol}..."):
        live_price, baselines, vol_data, err = fetch_market_and_vol_data(
            ticker_symbol, bypass_cache=force_refresh
        )

    if err:
        st.error(err)
    else:
        # Top KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Live Spot", f"${live_price:.2f}")
        kpi2.metric("Prior Day Close", f"${baselines['Daily']:.2f}")
        kpi3.metric("Weekly IV (Annual)", f"{vol_data['Weekly']['annual_iv'] * 100:.2f}%")
        kpi4.metric("Monthly IV (Annual)", f"{vol_data['Monthly']['annual_iv'] * 100:.2f}%")

        # Tab Navigation
        tab_main, tab_gauge, tab_term = st.tabs(
            ["📋 Expected Move Matrix", "🎯 Visual Sigma Gauge", "⏳ Term Structure Details"]
        )

        # Precompute bounds for calculations
        d_close = baselines["Daily"]
        d_sigma1 = d_close * vol_data["Daily"]["period_iv"]
        daily_bands = {
            "-3s": d_close - (3 * d_sigma1),
            "-2s": d_close - (2 * d_sigma1),
            "-1s": d_close - d_sigma1,
            "close": d_close,
            "+1s": d_close + d_sigma1,
            "+2s": d_close + (2 * d_sigma1),
            "+3s": d_close + (3 * d_sigma1),
        }

        # -----------------------------------------------------
        # TAB 1: Summary Matrix & Quick Status
        # -----------------------------------------------------
        with tab_main:
            # Inline Status Notification
            if daily_bands["-1s"] <= live_price <= daily_bands["+1s"]:
                msg = f"Live Price (${live_price:.2f}) is **INSIDE** the Daily 1σ range (${daily_bands['-1s']:.2f} to ${daily_bands['+1s']:.2f}) from prior close (${d_close:.2f})."
                st.success(msg)
            else:
                diff = (
                    (live_price - daily_bands["+1s"])
                    if live_price > daily_bands["+1s"]
                    else (live_price - daily_bands["-1s"])
                )
                msg = f"Live Price (${live_price:.2f}) is **OUTSIDE** the Daily 1σ range (${daily_bands['-1s']:.2f} to ${daily_bands['+1s']:.2f}) by ${abs(diff):.2f}."
                st.warning(msg)

            horizons = [
                ("Daily (1 Trading Day)", "Daily"),
                ("Weekly (Next Expiry)", "Weekly"),
                ("Monthly (~30-Day Expiry)", "Monthly"),
            ]

            records = []
            for label, key in horizons:
                anchor_close = baselines[key]
                p_pct = vol_data[key]["period_iv"]
                a_pct = vol_data[key]["annual_iv"]
                exp_date = vol_data[key]["exp"]
                dte = vol_data[key]["dte"]

                s1 = anchor_close * p_pct
                s2 = s1 * 2
                s3 = s1 * 3

                records.append(
                    {
                        "Horizon": label,
                        "Expiration": f"{exp_date} ({dte}d)",
                        "Annual IV": f"{a_pct * 100:.2f}%",
                        "Period Move": f"±{p_pct * 100:.2f}%",
                        "Anchor Close": f"${anchor_close:.2f}",
                        "1σ Range (68.3%)": f"${anchor_close - s1:.2f} — ${anchor_close + s1:.2f}",
                        "2σ Range (95.4%)": f"${anchor_close - s2:.2f} — ${anchor_close + s2:.2f}",
                        "3σ Range (99.7%)": f"${anchor_close - s3:.2f} — ${anchor_close + s3:.2f}",
                    }
                )

            df_display = pd.DataFrame(records)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # -----------------------------------------------------
        # TAB 2: Visual Plotly Sigma Gauge
        # -----------------------------------------------------
        with tab_gauge:
            st.subheader("Daily Range Deviation Gauge (-3σ to +3σ)")
            st.caption(
                f"Reference: Prior Close **${d_close:.2f}** | 1σ Move = **±${d_sigma1:.2f}**"
            )

            # Build Plotly Gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=live_price,
                    number={"prefix": "$", "valueformat": ".2f"},
                    gauge={
                        "axis": {
                            "range": [daily_bands["-3s"], daily_bands["+3s"]],
                            "tickwidth": 1,
                            "tickcolor": "#94a3b8",
                        },
                        "bar": {"color": "#38bdf8", "thickness": 0.25},
                        "bgcolor": "#1e293b",
                        "steps": [
                            {"range": [daily_bands["-3s"], daily_bands["-2s"]], "color": "#7f1d1d"},
                            {"range": [daily_bands["-2s"], daily_bands["-1s"]], "color": "#854d0e"},
                            {"range": [daily_bands["-1s"], daily_bands["+1s"]], "color": "#14532d"},
                            {"range": [daily_bands["+1s"], daily_bands["+2s"]], "color": "#854d0e"},
                            {"range": [daily_bands["+2s"], daily_bands["+3s"]], "color": "#7f1d1d"},
                        ],
                        "threshold": {
                            "line": {"color": "#f8fafc", "width": 4},
                            "thickness": 0.85,
                            "value": d_close,
                        },
                    },
                )
            )

            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f8fafc", "family": "Inter, sans-serif"},
            )
            st.plotly_chart(fig, use_container_width=True)

            g_col1, g_col2, g_col3 = st.columns(3)
            g_col1.metric("Lower 1σ Support", f"${daily_bands['-1s']:.2f}")
            g_col2.metric("Prior Close Center", f"${d_close:.2f}")
            g_col3.metric("Upper 1σ Resistance", f"${daily_bands['+1s']:.2f}")

        # -----------------------------------------------------
        # TAB 3: Term Structure Details
        # -----------------------------------------------------
        with tab_term:
            st.subheader("Selected Expiration Cycles")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.write("**Weekly Cycle Benchmark:**")
                st.write(f"- Contract Date: `{vol_data['Weekly']['exp']}`")
                st.write(f"- Days to Expiration: `{vol_data['Weekly']['dte']} days`")
                st.write(f"- Annualized Expiry IV: `{vol_data['Weekly']['annual_iv'] * 100:.2f}%`")
            with t_col2:
                st.write("**Monthly Cycle Benchmark:**")
                st.write(f"- Contract Date: `{vol_data['Monthly']['exp']}`")
                st.write(f"- Days to Expiration: `{vol_data['Monthly']['dte']} days`")
                st.write(f"- Annualized Expiry IV: `{vol_data['Monthly']['annual_iv'] * 100:.2f}%`")