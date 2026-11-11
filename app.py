import datetime
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Live IV & Expected Move", layout="wide")
st.title("📈 Real-Time Implied Volatility & Expected Move Calculator")

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
def fetch_realtime_data(symbol: str):
    tk = yf.Ticker(symbol)

    # 1. Fetch Current Price
    fast_info = getattr(tk, "fast_info", None)
    price = fast_info.last_price if fast_info else None
    if not price:
        hist = tk.history(period="5d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])

    if not price:
        return None, None, None, "Invalid ticker or no price data found."

    # 2. Select an Expiration Around 25-45 Days (Standard 30-day IV Benchmark)
    today = datetime.date.today()
    target_exp = None
    expirations = tk.options

    if not expirations:
        return None, None, None, f"No options chains available for {symbol}."

    # Find the expiration closest to 30 days out
    best_diff = 999
    for exp_str in expirations:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if 14 <= dte <= 50:  # Avoid 0-DTE and extreme long term
            diff = abs(dte - 30)
            if diff < best_diff:
                best_diff = diff
                target_exp = exp_str

    # Fallback to the first available if none in 14-50 range
    if not target_exp:
        target_exp = expirations[0]

    # 3. Pull Chain and Blend ATM Call + Put IV
    chain = tk.option_chain(target_exp)
    calls = chain.calls.dropna(subset=["strike", "impliedVolatility"])
    puts = chain.puts.dropna(subset=["strike", "impliedVolatility"])

    # Find ATM Call
    closest_call_idx = (calls["strike"] - price).abs().idxmin()
    call_iv = float(calls.loc[closest_call_idx, "impliedVolatility"])

    # Find ATM Put
    closest_put_idx = (puts["strike"] - price).abs().idxmin()
    put_iv = float(puts.loc[closest_put_idx, "impliedVolatility"])

    # Filter out bad ticks (e.g. 0.0001 or > 500%)
    valid_ivs = [v for v in [call_iv, put_iv] if 0.05 < v < 5.0]

    if valid_ivs:
        blended_iv = sum(valid_ivs) / len(valid_ivs)
        source = f"Blended ATM Call & Put ({target_exp})"
    else:
        # Fallback to 30-day realized volatility
        hist_30 = tk.history(period="1mo")
        log_rets = np.log(hist_30["Close"] / hist_30["Close"].shift(1))
        blended_iv = float(log_rets.std() * np.sqrt(252))
        source = "30-Day Realized Volatility (Fallback)"

    return price, blended_iv, source, None


if ticker_symbol:
    with st.spinner(f"Pulling live options chain for {ticker_symbol}..."):
        price, iv, source, err = fetch_realtime_data(ticker_symbol)

    if err:
        st.error(err)
    else:
        # Display top metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Live Spot Price", f"${price:.2f}")
        m2.metric("Annualized Implied Volatility", f"{iv * 100:.2f}%")
        m3.info(f"Source: {source}")

        st.markdown("---")
        st.subheader("Expected Move Matrix (1σ, 2σ, 3σ)")

        horizons = {
            "Daily": 1 / 252,
            "Weekly (5 Trading Days)": 5 / 252,
            "Monthly (21 Trading Days)": 21 / 252,
        }

        records = []
        for name, t_factor in horizons.items():
            sigma_1 = price * iv * math.sqrt(t_factor)
            sigma_2 = sigma_1 * 2
            sigma_3 = sigma_1 * 3
            period_pct = iv * math.sqrt(t_factor) * 100

            records.append(
                {
                    "Time Horizon": name,
                    "Expected % Move": f"±{period_pct:.2f}%",
                    "1σ Dollar Move": f"±${sigma_1:.2f}",
                    "1σ Range (68.3%)": f"${price - sigma_1:.2f} — ${price + sigma_1:.2f}",
                    "2σ Range (95.4%)": f"${price - sigma_2:.2f} — ${price + sigma_2:.2f}",
                    "3σ Range (99.7%)": f"${price - sigma_3:.2f} — ${price + sigma_3:.2f}",
                }
            )

        df_display = pd.DataFrame(records)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Custom expiration slider
        st.markdown("---")
        st.subheader("Custom Expiration Target (e.g. Earnings Date)")
        custom_days = st.slider("Days to Expiration (DTE):", 1, 90, 7)
        c_sigma1 = price * iv * math.sqrt(custom_days / 365)
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric(
            f"1σ Move ({custom_days} DTE)",
            f"±${c_sigma1:.2f}",
            f"Range: ${price - c_sigma1:.2f} - ${price + c_sigma1:.2f}",
        )
        c_col2.metric(
            f"2σ Move ({custom_days} DTE)",
            f"±${c_sigma1 * 2:.2f}",
            f"Range: ${price - c_sigma1 * 2:.2f} - ${price + c_sigma1 * 2:.2f}",
        )
        c_col3.metric(
            f"3σ Move ({custom_days} DTE)",
            f"±${c_sigma1 * 3:.2f}",
            f"Range: ${price - c_sigma1 * 3:.2f} - ${price + c_sigma1 * 3:.2f}",
        )