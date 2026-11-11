# -----------------------------------------------------
            # Daily 1σ Status Banner (Clean Inside / % Outside)
            # -----------------------------------------------------
            d_close = baselines["Daily"]
            d_sigma1 = d_close * vol_data["Daily"]["period_iv"]
            lower_1s = d_close - d_sigma1
            upper_1s = d_close + d_sigma1

            if lower_1s <= live_price <= upper_1s:
                st.success("Live price is currently inside the Daily 1σ implied range.")
            elif live_price > upper_1s:
                pct_above = ((live_price - upper_1s) / upper_1s) * 100
                st.warning(
                    f"Live price is currently **{pct_above:.2f}% above** the Daily 1σ upper band (${upper_1s:.2f})."
                )
            else:
                pct_below = ((lower_1s - live_price) / lower_1s) * 100
                st.warning(
                    f"Live price is currently **{pct_below:.2f}% below** the Daily 1σ lower band (${lower_1s:.2f})."
                )