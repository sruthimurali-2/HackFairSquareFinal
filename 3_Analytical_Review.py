import streamlit as st
from openai import AzureOpenAI
import os
import pandas as pd

st.set_page_config(page_title="Analytical Review - Augur", layout="centered")
st.title("Analytical Review")

st.subheader("Rationale Explanation Based on Model Prediction & Risk Factors")

model_level = st.session_state.get("model_pred", "N/A")
rf_level = st.session_state.get("rf_level", "N/A")
final_level = st.session_state.get("final_level", "N/A")

col1, col2, col3 = st.columns(3)
col1.metric("Model Predicted Level", model_level)
col2.metric("Risk Factor Observability Level", rf_level)
col3.metric("Final Adjusted Level", final_level)

if st.button("Explain Model Prediction Rationale using LLM"):
    if all(k in st.session_state for k in ["ir_summary", "vol_summary", "model_pred"]):
        st.session_state["rat_done"] = True

        api_key = os.getenv("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY", None)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT", None)

        if not api_key or not endpoint:
            st.error("❌ OpenAI credentials are missing. Please check environment variables or Streamlit secrets.")
        else:
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )

        messages = [
            {"role": "system", "content": "You are a financial analyst. Provide a brief, clear explanation of the model's IFRS13 classification based on the following inputs."},
            {"role": "user", "content": (
                f"IR Delta Summary:\n{st.session_state['ir_summary']}\n\n"
                f"Vol Summary:\n{st.session_state['vol_summary']}\n\n"
                f"Model Prediction: {st.session_state['model_pred']}\n"
                f"Risk Factor Observability Level: {st.session_state['rf_level']}\n"
                f"Final Level: {st.session_state['final_level']}\n"
                f"Provide a short rationale (3-4 lines) on key drivers behind the classification.\n"
                f"Include confidence level. Clearly state if Risk Fator Observability is different from Model Prediction and Justify Final Level."
            )}
        ]

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", st.secrets.get("AZURE_OPENAI_MODEL")),
            messages=messages,
            temperature=0.3
        )

        st.session_state["rationale_text"] = response.choices[0].message.content
        st.rerun()
    else:
        st.warning("⚠️ Ensure both model inference and risk summaries are completed.")

if "rationale_text" in st.session_state:
    st.success("✅ Rationale Generated")
    st.markdown(f"**Explanation:**\n\n{st.session_state['rationale_text']}\n")

# --- IFRS13 Expert Section ---
st.markdown("---------------------------------------------------------------------------------------------------")
st.header("IFRS13 Expert Assistant")
st.markdown("Ask any question about IFRS 13 fair value classification standards:")

user_question = st.text_area("Enter your question below:", placeholder="e.g., What is the US GAAP equivalent of IFRS13?")

if st.button("Ask IFRS13 Expert"):
    if user_question.strip():
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY", None)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT", None)

        if not api_key or not endpoint:
            st.error("❌ OpenAI credentials are missing.")
        else:
            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )

            expert_messages = [
                {"role": "system", "content": "You are a highly knowledgeable IFRS13 financial reporting expert. Explain user questions with clarity and brevity."},
                {"role": "user", "content": user_question}
            ]

            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_MODEL", st.secrets.get("AZURE_OPENAI_MODEL")),
                messages=expert_messages,
                temperature=0.2
            )

            st.success("✅ Expert Response")
            st.markdown(f"**Answer:**\n\n{response.choices[0].message.content}")
    else:
        st.warning("⚠️ Please enter a question to continue.")

# --- Movement Over Time Section ---
st.markdown("---------------------------------------------------------------------------------------------------")
st.header("Movement Over Time (MoT) Analysis")

def perform_mot_analysis(df_dec, df_mar):
    df_compare = pd.merge(df_dec, df_mar, on="trade_id", suffixes=("_dec", "_mar"))

    df_compare["Movement Summary"] = df_compare.apply(
        lambda row: f"No change (Level {row['Predicted IFRS13 Level_mar']})"
        if row['Predicted IFRS13 Level_dec'] == row['Predicted IFRS13 Level_mar']
        else f"Moved from {row['Predicted IFRS13 Level_dec']} to {row['Predicted IFRS13 Level_mar']}", axis=1)

    def get_mot_reason(old_level, new_level):
        if old_level == new_level:
            return "No change in classification — inputs remain consistent."
        transition = (old_level, new_level)
        reasons = {
            ("Level 3", "Level 2"): "Trade maturity has now entered the observable range.",
            ("Level 3", "Level 1"): "Market data inputs have become fully observable.",
            ("Level 2", "Level 1"): "Trade is now based entirely on quoted prices in active markets.",
            ("Level 1", "Level 2"): "Partial loss of observability in market inputs.",
            ("Level 2", "Level 3"): "Key valuation inputs have become unobservable.",
            ("Level 1", "Level 3"): "Significant deterioration in input observability."
        }
        return reasons.get(transition, "IFRS13 level changed due to re-evaluation of input observability.")

    df_compare["MOT Commentary"] = df_compare.apply(
        lambda row: get_mot_reason(row["Predicted IFRS13 Level_dec"], row["Predicted IFRS13 Level_mar"]),
        axis=1
    )

    return df_compare

uploaded_dec = st.file_uploader("Upload Trade Dataset 1 (with trade_id)", type=["csv"], key="dec")
uploaded_mar = st.file_uploader("Upload Trade Dataset (with trade_id)", type=["csv"], key="mar")

if uploaded_dec and uploaded_mar:
    df_dec = pd.read_csv(uploaded_dec)
    df_mar = pd.read_csv(uploaded_mar)

    if "trade_id" in df_dec.columns and "trade_id" in df_mar.columns and \
       "Predicted IFRS13 Level" in df_dec.columns and "Predicted IFRS13 Level" in df_mar.columns:
        result_df = perform_mot_analysis(df_dec, df_mar)
        st.success("✅ Movement Over Time Analysis Completed")

        # Show annotation
        st.markdown("---")
        st.markdown("### Trade-Level Commentary on Fair Value Classification Movements")

        # Show table
        st.dataframe(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download MoT Analysis CSV", data=csv, file_name="mot_analysis_results.csv", mime="text/csv")
    else:
        st.error("❌ Both files must contain 'trade_id' and 'Predicted IFRS13 Level' columns.")
