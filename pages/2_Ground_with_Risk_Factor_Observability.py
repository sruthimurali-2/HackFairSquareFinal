import streamlit as st
import pandas as pd
import os
from Observability_Stress_Module import simulate_greeks, generate_trade_pv_and_risk_pvs, ir_delta_stress_test, vol_risk_stress_test

st.set_page_config(page_title="Risk Factor Testing", layout="wide")
st.title("Grounding Model Predictions with Risk Factor Observability")

st.markdown("""
We perform a quantitative overlay on model-predicted fair value classifications by applying **risk factor-based stress testing**.  

It isolates the contribution of each input—such as IR Delta, Vega, Vanna, and Volga—to the trade's present value (PV), and determines the proportion attributable to **unobservable inputs**.  

In accordance with **IFRS 13** Level determination criteria, if the unobservable PV exceeds defined materiality thresholds (e.g., >10%),  
the model classification is re-leveled to reflect reduced market observability, ensuring alignment with valuation governance and disclosure standards.
""")

# --- Load secrets from environment variables as fallback ---
def get_secret(key, default=""):
    return os.getenv(key, st.secrets.get(key, default))

# --- Trade Input ---
st.sidebar.header("Trade Details")
product_type = st.sidebar.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
notional = st.sidebar.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)
currency = st.sidebar.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
option_type = st.sidebar.selectbox("Option Type", ["Receiver", "Payer"])
strike = st.sidebar.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
expiry_tenor = st.sidebar.selectbox("Expiry Tenor (Y)", [2, 3, 5, 10])
maturity_tenor = st.sidebar.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])

trade = {
    "product_type": product_type,
    "currency": currency,
    "option_type": option_type,
    "expiry_tenor": expiry_tenor,
    "maturity_tenor": maturity_tenor,
    "strike": strike,
    "notional": notional
}
st.session_state["trade"] = trade

# --- Show Model Predicted Level if available ---
model_level = st.session_state.get("model_pred")
if model_level:
    st.info(f" Fair value Level Predicted by Model: **{model_level}**")

if st.button("Ground with Risk Factor Observability"):
    greeks = simulate_greeks(trade)
    trade["trade_pv"], generated_pvs = generate_trade_pv_and_risk_pvs(greeks)
    greeks.update(generated_pvs)

    ir_stressed, ir_report, ir_stress_pv, ir_msgs = ir_delta_stress_test(trade, greeks)
    vol_stressed, vol_report, vol_stress_pv, vol_msgs = vol_risk_stress_test(trade, greeks)
    total_stress_pv = ir_stress_pv + vol_stress_pv
    rf_level = "Level 3" if total_stress_pv > 0.1 * trade["trade_pv"] else "Level 2"

    final_level = rf_level  #extend this logic for further adjustments

    st.session_state.update({
        "greeks": greeks,
        "generated_pvs": generated_pvs,
        "ir_summary": ir_msgs,
        "vol_summary": vol_msgs,
        "ir_report_df": pd.DataFrame(ir_report).T,
        "vol_report_df": pd.DataFrame(vol_report).T,
        "trade_pv": trade["trade_pv"],
        "ir_stress_pv": ir_stress_pv,
        "vol_stress_pv": vol_stress_pv,
        "rf_level": rf_level,
        "final_level": final_level,
        "rf_done": True
    })
    st.rerun()

# --- Display Results ---
if st.session_state.get("rf_done"):
    st.subheader("Classification Summary")

    model_level = st.session_state.get("model_pred", "N/A")
    rf_level = st.session_state.get("rf_level", "N/A")
    final_level = st.session_state.get("final_level", "N/A")
    observability_override = "Yes" if model_level != final_level else "No"

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Predicted Level", model_level)
    col2.metric("Risk Factor Observability Level", rf_level)
    col3.metric("Final Adjusted Level", final_level)

    st.markdown("---")
    st.subheader("Risk Factors and PV Contributions")
    st.dataframe(pd.DataFrame.from_dict(st.session_state["greeks"], orient="index", columns=["Value"]))
    st.dataframe(pd.DataFrame.from_dict(st.session_state["generated_pvs"], orient="index", columns=["PV"]))

    st.subheader("📈 IR Delta Observability")
    st.dataframe(st.session_state["ir_report_df"])

    st.subheader("📉 Volatility Observability")
    st.dataframe(st.session_state["vol_report_df"])

    total_pv = round(st.session_state["trade_pv"], 2)
    stress_pv = round(st.session_state["ir_stress_pv"] + st.session_state["vol_stress_pv"], 2)
    st.metric("Total PV", total_pv)
    st.metric("Stressed PV", stress_pv)

    st.success(f"🔍 Final Observability Level: {final_level}")