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
    selected_preset = None
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
    """Fetches live stock price and near-the-money options implied volatility."""
    tk = yf.Ticker(symbol)

    # 1. Spot Price
    price = None
    fast_info = getattr(tk, "fast_info", None)
    if fast_info and hasattr(fast_info, "last_price"):
        price = fast_info.last_price
    if not price:
        hist = tk.history(period="5d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])

    if not price:
        return None, None, None, "Invalid ticker or no price data found."

    # 2. Options Implied Volatility
    iv = None
    expiry_used = None
    try:
        expirations = tk.options
        if expirations:
            # Pick first expiration >= 5 days out to avoid 0-DTE pricing skews
            expiry_used = expirations[0]
            for exp in expirations:
                chain = tk.option_chain(exp)
                calls = chain.calls.dropna(
                    subset=["impliedVolatility", "strike"]
                )
                if not calls.empty:
                    closest_idx = (calls["strike"] - price).abs().idxmin()
                    candidate_iv = float(
                        calls.loc[closest_idx, "impliedVolatility"]
                    )
                    if candidate_iv > 0.02:
                        iv = candidate_iv
                        expiry_used = exp
                        break
    except Exception:
        pass

    # 3. Fallback to 30-day realized volatility if options chain is empty/unavailable
    vol_source = f"ATM Call ({expiry_used} Expiration)"
    if iv is None or iv <= 0.01:
        hist_30 = tk.history(period="1mo")
        log_rets = np.log(hist_30["Close"] / hist_30["Close"].shift(1))
        iv = float(log_rets.std() * np.sqrt(252))
        vol_source = "30-Day Realized Volatility (Options chain unavailable)"

    return price, iv, vol_source, None


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