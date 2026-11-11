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
ticker_symbol = st.text_input("Enter Ticker Symbol:", value=default_ticker).strip().upper()


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
    if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price is not None:
        live_price = float(fast_info.last_price)
    else:
        live_price = float(hist_daily["Close"].iloc[-1])

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
        "Monthly": prev_month_close
    }

    # 2. Match Expirations for Weekly (~7 DTE) and Monthly (~30 DTE)
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

    # Fallbacks if options data is sparse
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
            "dte": exp_dtes.get(weekly_exp, 1)
        },
        "Weekly": {
            "annual_iv": weekly_iv,
            "period_iv": weekly_iv * math.sqrt(exp_dtes.get(weekly_exp, 7) / 365),
            "exp": weekly_exp,
            "dte": exp_dtes.get(weekly_exp, 7)
        },
        "Monthly": {
            "annual_iv": monthly_iv,
            "period_iv": monthly_iv * math.sqrt(exp_dtes.get(monthly_exp, 30) / 365),
            "exp": monthly_exp,
            "dte": exp_dtes.get(monthly_exp, 30)
        }
    }

    return live_price, baselines, vol_data, None


if ticker_symbol:
    with st.spinner(f"Pulling term-structure options data for {ticker_symbol}..."):
        live_price, baselines, vol_data, err = fetch_market_and_vol_data(ticker_symbol)

    if err:
        st.error(err)
    else:
        # Top summary cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Spot Price", f"${live_price:.2f}")
        m2.metric("Prior Day Close", f"${baselines['Daily']:.2f}")
        m3.metric("Weekly Expiry IV (Annual)", f"{vol_data['Weekly']['annual_iv'] * 100:.2f}%")
        m4.metric("Monthly Expiry IV (Annual)", f"{vol_data['Monthly']['annual_iv'] * 100:.2f}%")

        st.markdown("---")
        st.subheader("Expected Move Bands (Centered on Prior Period Close)")

        horizons = [
            ("Daily (1 Trading Day)", "Daily"),
            ("Weekly (Next Expiration)", "Weekly"),
            ("Monthly (~30-Day Expiry)", "Monthly"),
        ]

        records = []
        for label, key in horizons:
            anchor_close = baselines[key]
            period_pct = vol_data[key]["period_iv"]
            annual_pct = vol_data[key]["annual_iv"]
            exp_date = vol_data[key]["exp"]
            dte = vol_data[key]["dte"]

            sigma_1 = anchor_close * period_pct
            sigma_2 = sigma_1 * 2
            sigma_3 = sigma_1 * 3

            records.append({
                "Horizon": label,
                "Expiration Used": f"{exp_date} ({dte} DTE)",
                "Annual IV": f"{annual_pct * 100:.2f}%",
                "Period Move %": f"±{period_pct * 100:.2f}%",
                "Anchor Close": f"${anchor_close:.2f}",
                "1σ Range (68.3%)": f"${anchor_close - sigma_1:.2f} —${anchor_close + sigma_1:.2f}",
                "2σ Range (95.4%)": f"${anchor_close - sigma_2:.2f} —${anchor_close + sigma_2:.2f}",
                "3σ Range (99.7%)": f"${anchor_close - sigma_3:.2f} —${anchor_close + sigma_3:.2f}",
            })

        df_display = pd.DataFrame(records)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Session tracking indicator
        d_close = baselines["Daily"]
        d_sigma1 = d_close * vol_data["Daily"]["period_iv"]
        lower_1s = d_close - d_sigma1
        upper_1s = d_close + d_sigma1

        st.markdown("---")
        st.write("**Current Session Position vs. Daily 1σ Envelope:**")
        if lower_1s <= live_price <= upper_1s:
            st.success(
                f"Live Price (${live_price:.2f}) is **INSIDE** the Daily 1σ range "
                f"(${lower_1s:.2f} to ${upper_1s:.2f}) from yesterday's close (${d_close:.2f})."
            )
        else:
            diff = live_price - upper_1s if live_price > upper_1s else live_price - lower_1s
            st.warning(
                f"Live Price (${live_price:.2f}) is **OUTSIDE** the Daily 1σ range "