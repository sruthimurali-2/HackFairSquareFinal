import streamlit as st
import pandas as pd
import os
import requests
import time
from openai import AzureOpenAI
from streamlit_echarts import st_echarts

def predict_ir_swaption(input_df):
        try:
            input_df = input_df.applymap(lambda x: None if pd.isna(x) or x in [float("inf"), float("-inf")] else x)
            payload = {
                "input_data": {
                    "columns": input_df.columns.tolist(),
                    "index": [0],
                    "data": input_df.values.tolist()
                }
            }

            start_time = time.time()
            response = requests.post(
                url=get_secret("AZURE_ML_ENDPOINT"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {get_secret('AZURE_ML_API_KEY')}"
                },
                json=payload
            )
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)

            response.raise_for_status()
            result = response.json()
            return result[0] if isinstance(result, list) else result

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Model call failed: {e}")
            return {"error": str(e)}

# --- Mock model predictors per product ---
def predict_bond(input_data):
    try:
        if "ISIN" not in input_data.columns:
            return "Missing ISIN in input"

        isin = input_data["ISIN"].values[0]
        bond_df = pd.read_csv("bond_observability.csv")

        # Normalize column names
        bond_df.columns = bond_df.columns.str.strip().str.lower()

        # Expect column to be named something like 'observable_level'
        if "observable_level" not in bond_df.columns:
            return "Column 'observable_level' not found in CSV"

        match = bond_df[bond_df["isin"] == isin]
        if not match.empty:
            return match["observable_level"].values[0]
        else:
            return f"Unknown ISIN: {isin}"
    except Exception as e:
        return f"Error: {e}"

def predict_capfloor(input_data):
    return "Level 3"

def predict_irswap(input_data):
    return "Level 2"

def predict_by_product(product_type, input_data):
    if product_type == "IR Swaption":
        return predict_ir_swaption(input_data)
    elif product_type == "Bond":
        return predict_bond(input_data)
    elif product_type == "CapFloor":
        return predict_capfloor(input_data)
    elif product_type == "IRSwap":
        return predict_irswap(input_data)
    else:
        return "Unknown"

def get_secret(key, default=""):
    try:
        return os.getenv(key) or st.secrets.get(key, default)
    except Exception:
        return default

# --- Streamlit App Layout ---
st.set_page_config(page_title="Augur - Fair Value Classification Model",layout="centered")

st.title("Augur - Fair Value Classification Model")

# --- Workflow 
single_tab, batch_tab = st.tabs(["Predict Fair value Level for a Single trade", "Batch Prediction" ])

with single_tab:
    st.subheader("Predict Fair Value Classification using Machine Learning Model")
    with st.sidebar:
        st.markdown("### Trade Details")
        product_type = st.selectbox("Product Type", ["IR Swaption", "Bond", "CapFloor", "IRSwap"], index=0)
        st.session_state["product_type"] = product_type

        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY"])
        notional = st.number_input("Notional", min_value=1_000_000, step=1_000_000, value=10_000_000)

        if product_type in ["IR Swaption", "CapFloor", "IRSwap"]:
            option_type = st.selectbox("Option Type", ["Receiver", "Payer"])
            strike = st.slider("Strike (%)", 0.0, 10.0, 2.5, 0.1)
            expiry_tenor = st.selectbox("Expiry Tenor (Y)", [2, 3, 5, 10])
            maturity_tenor = st.selectbox("Maturity Tenor (Y)", [5, 10, 15, 20, 30])
        elif product_type == "Bond":
            option_type = "N/A"
            strike = 0.0
            expiry_tenor = 0
            maturity_tenor = st.selectbox("Maturity Tenor (Y)", [1, 2, 3, 5, 10, 30])
            rating = st.selectbox("Credit Rating", ["AAA", "AA", "A", "BBB", "BB", "B"])
            issuer_type = st.selectbox("Issuer Type", ["Government", "Corporate"])

            # Load ISIN list from CSV
            try:
                bond_obs_df = pd.read_csv("bond_observability.csv")
                isin_list = bond_obs_df["ISIN"].dropna().unique().tolist()
                isin = st.selectbox("ISIN", isin_list)
            except Exception as e:
                isin = st.text_input("ISIN (CSV load failed, enter manually)", "")
                st.error(f"Could not load ISIN list: {e}")

            # trade_inputs.update({
            #     "rating": rating,
            #     "issuer_type": issuer_type,
            #     "ISIN": isin
            # })


        trade_inputs = {
            "product_type": product_type,
            "ISIN": isin if product_type == "Bond" else "",
            "currency": currency,
            "notional": notional,
            "option_type": option_type,
            "strike": strike,
            "expiry_tenor": expiry_tenor,
            "maturity_tenor": maturity_tenor,
        }

        if product_type == "Bond":
            trade_inputs.update({
                "rating": rating,
                "issuer_type": issuer_type
            })

    input_data = pd.DataFrame([trade_inputs])

    if st.button("Run Single Trade Inference"):
        try:
            start_time = time.time()
            result = predict_by_product(product_type, input_data)
            end_time = time.time()
            st.session_state["model_pred"] = result
            st.session_state["ML_Model_elapsed_time"] = round(end_time - start_time, 2)
            
            with st.expander("Model Input", expanded=False):
                st.subheader("📦 Input JSON Payload")
                st.json(input_data.to_dict(orient="records")[0])

            with st.expander("Model Details", expanded=False):
                st.markdown(f"⏱️ Model run completed in {st.session_state['ML_Model_elapsed_time']} seconds")

                if st.session_state.get("product_type") == "Bond":
                    st.markdown("""
                        <div style='text-align: left; padding: 10px; background-color: #eeeeee; border-radius: 8px; color: #000000; font-family: monospace; font-size: 14px;'>
                        <strong>Model:</strong> ISIN Lookup based Classification<br>
                        <strong>Version:</strong> Bond Market Observability (Static Lookup)<br>
                        <strong>Method:</strong> Classification based on Observable Market Inputs (ISIN)<br>
                        <strong>Source:</strong> bond_observability.csv<br>
                        <strong>Features:</strong> ISIN<br>
                        <strong>Description:</strong> Predicted Fair Value Level for traded bonds using Market Observability derived from listed ISIN and its classification.<br>
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown("""
                        <div style='text-align: left; padding: 10px; background-color: #eeeeee; border-radius: 8px; color: #000000; font-family: monospace; font-size: 14px;'>
                        <strong>Model:</strong> Gradient Boosting (AutoML)<br>
                        <strong>Version:</strong> Gradient Boosting (AutoML)<br>
                        <strong>Trained on:</strong> Synthetic IR Swaption Trades<br>
                        <strong>Features:</strong> product_type, currency, option_type, notional, strike, expiry_tenor, maturity_tenor<br>
                        <strong>Accuracy:</strong> 86.2%<br>
                        <strong>AUC:</strong> 0.74<br>
                        </div>
                    """, unsafe_allow_html=True)
                      
                
            st.success(f"✅ Predicted Fair Value Classification: {result}")
            st.markdown("""
                        ⚠️ **Responsible AI Notice:** Fair value level predicted by Model is for internal reporting only and should not be used for external or regulatory disclosure.
                        Please use Augmented Prediction for external reporting.
                        """)
        except Exception as e:
            st.error(f"❌ Mock model failed: {e}")


# --- Batch Inference and Analytical Review ---

with batch_tab:
    st.subheader("Predict Fair Value Classification for a set of trades")
    st.markdown("Upload a CSV file with trade deatils - Supported Products - IR Swaption, Bond, CapFloor, IRSwap")

    uploaded_file = st.file_uploader("Upload CSV", type="csv", key="batch")
    if uploaded_file:
        df_infer = pd.read_csv(uploaded_file)
        required_cols = ["product_type", "currency", "option_type", "notional", "strike", "expiry_tenor", "maturity_tenor", "ISIN"]

        if all(col in df_infer.columns for col in required_cols):
            with st.spinner("Running batch inference..."):
                try:
                    df_infer["Predicted IFRS13 Level"] = df_infer.apply(
                        lambda row: predict_by_product(row["product_type"], row.to_frame().T), axis=1
                    )
                    st.success("✅ Inference completed!")
                    st.dataframe(df_infer.head(11))

                    st.download_button("📅 Download Results", data=df_infer.to_csv(index=False), file_name="predicted_results.csv")
                except Exception as e:
                    st.error(f"❌ Batch mock model failed: {e}")
        else:
            st.warning(f"CSV must include columns: {', '.join(required_cols)}")

                   # --- Development-only Visualization ---
        if "trading_desk" in df_infer.columns:
                        heatmap_data = df_infer.groupby(["trading_desk", "Predicted IFRS13 Level"]).size().reset_index(name="count")
                        rows = heatmap_data["trading_desk"].unique().tolist()
                        cols = ["Level 1", "Level 2", "Level 3"]
                        df_infer = df_infer[df_infer["Predicted IFRS13 Level"].isin(cols)]
                        
                        row_map = {v: i for i, v in enumerate(rows)}
                        col_map = {v: i for i, v in enumerate(cols)}

                        data = [[col_map[c], row_map[r], int(v)] for r, c, v in heatmap_data.values]

                        option = {
                            "tooltip": {"position": "top"},
                            "grid": {"height": "50%", "top": "10%", "left": "30%"},
                            "xAxis": {"type": "category", "data": cols, "splitArea": {"show": True}},
                            "yAxis": {"type": "category", "data": rows, "splitArea": {"show": True}},
                            "visualMap": {
                                "min": 0,
                                "max": max(heatmap_data["count"]),
                                "calculable": True,
                                "orient": "horizontal",
                                "left": "center",
                                "bottom": "15%",
                            },
                            "series": [
                                {
                                    "name": "Trade Count",
                                    "type": "heatmap",
                                    "data": data,
                                    "label": {"show": True},
                                    "emphasis": {
                                        "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                                    },
                                }
                            ],
                        }

                        st.subheader("Heatmap: Predicted Fair value Level by Trading Desk")
                        st_echarts(option, height="400px")
        else:
                    st.warning(f"CSV must include columns: {', '.join(required_cols)}")
        
        if "product_type" in df_infer.columns and "Predicted IFRS13 Level" in df_infer.columns:
            heatmap_data_prod = df_infer.groupby(["product_type", "Predicted IFRS13 Level"]).size().reset_index(name="count")
            rows = heatmap_data_prod["product_type"].unique().tolist()
            cols = heatmap_data_prod["Predicted IFRS13 Level"].unique().tolist()

            row_map = {v: i for i, v in enumerate(rows)}
            col_map = {v: i for i, v in enumerate(cols)}

            data = [[col_map[c], row_map[r], int(v)] for r, c, v in heatmap_data_prod.values]

            option_prod = {
                "tooltip": {"position": "top"},
                "grid": {"height": "50%", "top": "10%", "left": "30%"},
                "xAxis": {"type": "category", "data": cols, "splitArea": {"show": True}},
                "yAxis": {"type": "category", "data": rows, "splitArea": {"show": True}},
                "visualMap": {
                    "min": 0,
                    "max": max(heatmap_data_prod["count"]),
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": "15%",
                },
                "series": [
                    {
                        "name": "Trade Count",
                        "type": "heatmap",
                        "data": data,
                        "label": {"show": True},
                        "emphasis": {
                            "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                        },
                    }
                ],
            }

            st.subheader("Heatmap: Predicted Fair Value Level by Product Type")
            st_echarts(option_prod, height="400px")


# with rationale_tab:
#     st.subheader("Analytical review")
#     if st.button("Run GPT-4o Rationale"):
#         if all(k in st.session_state for k in ["ir_summary", "vol_summary", "model_pred"]):
#             st.session_state["rat_done"] = True
#             client = AzureOpenAI(
#                 api_key=get_secret("AZURE_OPENAI_API_KEY"),
#                 api_version="2024-02-01",
#                 azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT")
#             )
#             messages = [
#                 {"role": "system", "content": "You're a financial analyst..."},
#                 {"role": "user", "content": (
#                     f"IR Delta Summary:\n{st.session_state['ir_summary']}\n\n"
#                     f"Vol Summary:\n{st.session_state['vol_summary']}\n\n"
#                     f"Model Prediction: {st.session_state['model_pred']}\n"
#                     "Explain and confirm IFRS13 classification with confidence score."
#                 )}
#             ]
#             response = client.chat.completions.create(
#                 model=get_secret("AZURE_OPENAI_MODEL"),
#                 messages=messages,
#                 temperature=0.5
#             )
#             st.session_state["rationale_text"] = response.choices[0].message.content
#             st.rerun()
#         else:
#             st.warning("⚠️ Ensure both model inference and risk summaries are completed.")

#     if "rationale_text" in st.session_state:
#         st.success("✅ Rationale Generated")
#         st.markdown(f"**Explanation:**\n\n{st.session_state['rationale_text']}")
