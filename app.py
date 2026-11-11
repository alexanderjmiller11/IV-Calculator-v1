import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live IV & Expected Move", layout="wide")
st.title("📈 Expected Move Calculator (Anchored to Prior Period Closes)")

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


@st.cache_data(ttl=60)
def fetch_market_and_vol_data(symbol: str):
    tk = yf.Ticker(symbol)

    # 1. Fetch Current Live Price & Historical Baseline Closes
    hist_daily = tk.history(period="3mo")
    if hist_daily.empty:
        return None, None, None, None, f"No price history found for {symbol}."

    fast_info = getattr(tk, "fast_info", None)
    live_price = (
        fast_info.last_price
        if (fast_info and hasattr(fast_info, "last_price"))
        else float(hist_daily["Close"].iloc[-1])
    )

    # Daily baseline: Prior trading day close
    # If market is open, iloc[-2] is yesterday's close; if closed/pre-market, adjust safely
    if len(hist_daily) >= 2:
        prev_day_close = float(hist_daily["Close"].iloc[-2])
    else:
        prev_day_close = float(hist_daily["Close"].iloc[-1])

    # Weekly baseline: Last week's closing price
    hist_weekly = tk.history(period="6mo", interval="1wk")
    if len(hist_weekly) >= 2:
        prev_week_close = float(hist_weekly["Close"].iloc[-2])
    else:
        prev_week_close = prev_day_close

    # Monthly baseline: Last month's closing price
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

    # 2. Select Expiration Around 25-45 Days Out for Normalized 30-day IV
    today = datetime.date.today()
    target_exp = None
    expirations = tk.options

    if not expirations:
        return None, None, None, None, f"No options chains available for {symbol}."

    best_diff = 999
    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if 14 <= dte <= 50:
            diff = abs(dte - 30)
            if diff < best_diff:
                best_diff = diff
                target_exp = exp_str

    if not target_exp:
        target_exp = expirations[0]

    # 3. Pull Chain and Blend ATM Call + Put IV
    chain = tk.option_chain(target_exp)
    calls = chain.calls.dropna(subset=["strike", "impliedVolatility"])
    puts = chain.puts.dropna(subset=["strike", "impliedVolatility"])

    closest_call_idx = (calls["strike"] - live_price).abs().idxmin()
    call_iv = float(calls.loc[closest_call_idx, "impliedVolatility"])

    closest_put_idx = (puts["strike"] - live_price).abs().idxmin()
    put_iv = float(puts.loc[closest_put_idx, "impliedVolatility"])

    valid_ivs = [v for v in [call_iv, put_iv] if 0.05 < v < 5.0]

    if valid_ivs:
        blended_iv = sum(valid_ivs) / len(valid_ivs)
        source = f"Blended ATM Call & Put ({target_exp})"
    else:
        log_rets = np.log(hist_daily["Close"] / hist_daily["Close"].shift(1))
        blended_iv = float(log_rets.std() * np.sqrt(252))
        source = "30-Day Realized Volatility (Fallback)"

    return live_price, baselines, blended_iv, source, None


if ticker_symbol:
    with st.spinner(f"Pulling market data and historical baselines for {ticker_symbol}..."):
        live_price, baselines, iv, source, err = fetch_market_and_vol_data(ticker_symbol)

    if err:
        st.error(err)
    else:
        # Top summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"${live_price:.2f}")
        m2.metric("Prior Day Close", f"${baselines['Daily']:.2f}")
        m3.metric("Annualized IV", f"{iv * 100:.2f}%")
        m4.info(f"IV Source: {source}")

        st.markdown("---")
        st.subheader("Expected Move Ranges (Centered on Prior Period Close)")

        time_horizons = [
            ("Daily (1 Trading Day)", baselines["Daily"], 1 / 252),
            ("Weekly (5 Trading Days)", baselines["Weekly"], 5 / 252),
            ("Monthly (21 Trading Days)", baselines["Monthly"], 21 / 252),
        ]

        records = []
        for name, anchor_close, t_factor in time_horizons:
            sigma_pct = iv * math.sqrt(t_factor)
            sigma_1 = anchor_close * sigma_pct
            sigma_2 = sigma_1 * 2
            sigma_3 = sigma_1 * 3

            records.append(
                {
                    "Horizon": name,
                    "Anchor Close": f"${anchor_close:.2f}",
                    "Expected %": f"±{sigma_pct * 100:.2f}%",
                    "1σ Move (±$)": f"±${sigma_1:.2f}",
                    "1σ Range (68.3%)": f"${anchor_close - sigma_1:.2f} — ${anchor_close + sigma_1:.2f}",
                    "2σ Range (95.4%)": f"${anchor_close - sigma_2:.2f} — ${anchor_close + sigma_2:.2f}",
                    "3σ Range (99.7%)": f"${anchor_close - sigma_3:.2f} — ${anchor_close + sigma_3:.2f}",
                }
            )

        df_display = pd.DataFrame(records)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Visual indicator showing where current price sits relative to daily range
        d_close = baselines["Daily"]
        d_sigma1 = d_close * (iv * math.sqrt(1 / 252))
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
                f"(${lower_1s:.2f} to ${upper_1s:.2f}) by ${abs(diff):.2f}."
            )