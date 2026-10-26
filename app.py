import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- App Configuration ---
st.set_page_config(
    page_title="Stockformer 2.0: Project Demo",
    layout="wide",
)

# --- Sidebar Navigation ---
st.sidebar.title("ML Project (25RK28)")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Data Exploration", "Training Curves", "Model Results"]
)

# ============================================================
# --- PAGE 1: PROJECT OVERVIEW ---
# ============================================================
if page == "Project Overview":
    st.title("Transformer-Based Financial Time-Series Forecasting")
    st.header("Project Overview and Methodology")

    st.markdown("""
    Welcome to **Stockformer 2.0**, a Transformer-based system for financial time-series forecasting.  
    This dashboard presents the results and methodology of your ML project.

    ### 🎯 Project Highlights:
    - **Advanced Feature Engineering:** RSI, MACD, ATR, Volume SMA  
    - **VMD-PatchTST Model:** Combines Variational Mode Decomposition + PatchTST  
    - **Asymmetric Loss Function:** Penalizes wrong-direction predictions more heavily for better risk-adjusted returns  

    Explore:
    - **Data Exploration** → visualize features  
    - **Training Curves** → view learning behavior  
    - **Model Results** → compare Stockformer vs PatchTST  
    """)

# ============================================================
# --- PAGE 2: DATA EXPLORATION ---
# ============================================================
elif page == "Data Exploration":
    st.title("Data Exploration and Feature Engineering")
    st.sidebar.header("Data Controls")
    ticker = st.sidebar.text_input("Enter Stock Ticker", "ICICIBANK.NS")

    if st.sidebar.button("Load Data"):
        data = yf.download(ticker, period="730d", interval="1h")

        if data.empty:
            st.error("No data returned for ticker. Please check the symbol or try another interval.")
        else:
            # --- FIX MULTI-INDEX ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]  # flatten

            st.subheader(f"📊 Raw Data for {ticker}")
            st.dataframe(data.tail())

            # --- Plot Price ---
            fig_price = px.line(data, y="Close", title=f"{ticker} Closing Price")
            st.plotly_chart(fig_price, width="stretch")

            # --- Feature Engineering ---
            data["log_return"] = np.log(data["Close"] / data["Open"])
            data["RSI_14"] = ta.rsi(data["Close"], length=14)
            data["ATR_14"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)
            data = data.dropna()

            st.subheader("🧩 Engineered Features (RSI, ATR, Log Returns)")
            st.dataframe(data[["log_return", "RSI_14", "ATR_14"]].tail())

            fig_rsi = px.line(data, y="RSI_14", title=f"{ticker} 14-Hour RSI")
            st.plotly_chart(fig_rsi, width="stretch")

# ============================================================
# --- PAGE 3: TRAINING CURVES ---
# ============================================================
elif page == "Training Curves":
    st.title("📉 Model Training Curves (Loss History)")
    st.markdown("""
    Visualizing training and validation loss for both **Stockformer** and **PatchTST** models.
    """)

    try:
        with open("reports/loss_history.json", "r") as f:
            history = json.load(f)

        df_stock = pd.DataFrame({
            "Epoch": range(1, len(history["stockformer"]["train"]) + 1),
            "Train Loss": history["stockformer"]["train"],
            "Val Loss": history["stockformer"]["val"],
            "Model": "Stockformer"
        })

        df_patch = pd.DataFrame({
            "Epoch": range(1, len(history["patchtst"]["train"]) + 1),
            "Train Loss": history["patchtst"]["train"],
            "Val Loss": history["patchtst"]["val"],
            "Model": "PatchTST"
        })

        df_all = pd.concat([df_stock, df_patch])

        st.subheader("📊 Training vs Validation Loss")
        fig_loss = px.line(
            df_all.melt(id_vars=["Epoch", "Model"], value_vars=["Train Loss", "Val Loss"]),
            x="Epoch", y="value", color="Model", line_dash="variable",
            title="Training & Validation Loss Curves"
        )
        fig_loss.update_layout(yaxis_title="Loss", xaxis_title="Epoch")
        st.plotly_chart(fig_loss, width="stretch")

    except FileNotFoundError:
        st.error("❌ `loss_history.json` not found. Please run `python src/models/train.py` first.")

# ============================================================
# --- PAGE 4: MODEL RESULTS ---
# ============================================================
elif page == "Model Results":
    st.title("Model Results and Demonstration")
    st.markdown("""
    Compare model forecasts and backtested performance on the unseen test set.
    """)

    results_path = "reports/test_results.csv"

    try:
        df = pd.read_csv(results_path, index_col=0, parse_dates=True)
        st.success(f"✅ Loaded model predictions from `{results_path}`")

        # --- Predicted vs Actual ---
        st.subheader("📈 Predicted vs Actual Returns")
        fig_pred = px.line(
            df,
            x=df.index,
            y=["Actual", "Stockformer_Pred", "PatchTST_Pred"],
            labels={"value": "Return"},
            title="Predicted vs Actual Returns (Test Set)"
        )
        st.plotly_chart(fig_pred, width="stretch")

        # --- Cumulative ROI ---
        df["Stockformer_Strategy"] = (1 + df["Actual"] * df["Stockformer_Pred"]).cumprod()
        df["PatchTST_Strategy"] = (1 + df["Actual"] * df["PatchTST_Pred"]).cumprod()
        df["BuyHold"] = (1 + df["Actual"]).cumprod()

        st.subheader("💰 Cumulative ROI Comparison")
        fig_roi = px.line(
            df,
            x=df.index,
            y=["BuyHold", "Stockformer_Strategy", "PatchTST_Strategy"],
            labels={"value": "Cumulative ROI"},
            title="Cumulative ROI vs Buy & Hold"
        )
        st.plotly_chart(fig_roi, width="stretch")

        # --- Compute Metrics ---
        def compute_metrics(pred_col):
            mse = mean_squared_error(df["Actual"], df[pred_col])
            mae = mean_absolute_error(df["Actual"], df[pred_col])
            strategy_returns = df["Actual"] * df[pred_col]
            sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9)
            roi = (1 + strategy_returns).cumprod().iloc[-1] - 1
            return mse, mae, roi * 100, sharpe

        metrics = {
            "Stockformer": compute_metrics("Stockformer_Pred"),
            "PatchTST": compute_metrics("PatchTST_Pred")
        }

        metrics_df = pd.DataFrame(metrics, index=["MSE", "MAE", "ROI (%)", "Sharpe Ratio"])
        st.subheader("📊 Performance Metrics")
        st.table(metrics_df.style.format("{:.6f}"))

        # --- ROI vs Sharpe Scatter ---
        st.subheader("📉 ROI vs Sharpe Trade-off")
        scatter_df = pd.DataFrame({
            "Model": ["Stockformer", "PatchTST"],
            "ROI (%)": [
                metrics_df.loc["ROI (%)", "Stockformer"],
                metrics_df.loc["ROI (%)", "PatchTST"]
            ],
            "Sharpe Ratio": [
                metrics_df.loc["Sharpe Ratio", "Stockformer"],
                metrics_df.loc["Sharpe Ratio", "PatchTST"]
            ]
        })

        scatter_df["Marker Size"] = scatter_df["ROI (%)"].abs() + 1e-3

        fig_scatter = px.scatter(
            scatter_df,
            x="ROI (%)",
            y="Sharpe Ratio",
            color="Model",
            size="Marker Size",
            text="Model",
            title="Risk-Adjusted Return Comparison"
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, width="stretch")

    except FileNotFoundError:
        st.error(" Results file not found. Please run `python src/models/predict.py` first.")
