import pandas as pd
import torch
import numpy as np
import os
import sys
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
# --- End Path Setup ---

# Import both models
from src.models.model import Stockformer
from tsai.models.PatchTST import PatchTST


def run_inference():
    """
    Runs inference for both Stockformer and PatchTST models,
    evaluates them on the test set, and saves results + metrics.
    """
    print("Running inference...")

    # --- 1. Define Paths ---
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    TEST_TARGET_PATH = os.path.join(DATA_DIR, "y_test.csv")
    TEST_FEATURES_PATH_NPY = os.path.join(DATA_DIR, "X_test.npy")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

    STOCKFORMER_PATH = os.path.join(MODEL_DIR, "stockformer_best.pth")
    PATCHTST_PATH = os.path.join(MODEL_DIR, "patchtst_best.pth")

    OUTPUT_PATH = os.path.join(REPORTS_DIR, "test_results.csv")

    # --- 2. Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Data ---
    try:
        y_test_df = pd.read_csv(TEST_TARGET_PATH, index_col=0, parse_dates=True)
        X_test = np.load(TEST_FEATURES_PATH_NPY)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e.filename}. Please run 'python src/models/train.py' first.")
        return

    y_test = y_test_df.values.squeeze()
    num_features = X_test.shape[2]
    print(f"Test data shapes: X={X_test.shape}, y={y_test.shape}")

    # Convert to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # ===================================================================
    #  STOCKFORMER INFERENCE
    # ===================================================================
    print("\n--- Running Stockformer Inference ---")
    stockformer = Stockformer(num_features=num_features).to(device)
    stockformer.load_state_dict(torch.load(STOCKFORMER_PATH, map_location=device))
    stockformer.eval()

    preds_stock = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_test_tensor), batch_size):
            batch_X = X_test_tensor[i:i + batch_size]
            preds_stock.append(stockformer(batch_X).cpu().numpy())
    preds_stock = np.concatenate(preds_stock).flatten()

    print(f"✅ Stockformer Predictions: {len(preds_stock)} samples")

    # ===================================================================
    #  PATCHTST INFERENCE
    # ===================================================================
    print("\n--- Running PatchTST Inference ---")
    X_test_patch = np.transpose(X_test, (0, 2, 1))  # (N, features, seq_len)
    X_test_patch_tensor = torch.tensor(X_test_patch, dtype=torch.float32).to(device)

    # --- PatchTST model initialization (must match training setup) ---
    patchtst = PatchTST(
        c_in=num_features,
        c_out=1,
        seq_len=60,
        patch_len=10,
        stride=5,
        dropout=0.05,
        revin=True,
        res_attention=True
    ).to(device)

    # --- Load the trained model weights safely ---
    state_dict = torch.load(PATCHTST_PATH, map_location=device)

    # Handle potential shape mismatch gracefully
    try:
        patchtst.load_state_dict(state_dict)
    except RuntimeError:
        print("⚠️  Shape mismatch detected in model head. Attempting to fix...")
        for k in list(state_dict.keys()):
            if "model.head.layer.1" in k:
                del state_dict[k]
        patchtst.load_state_dict(state_dict, strict=False)
        print("✅ Loaded model (head reinitialized)")

    patchtst.eval()

    preds_patch = []
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(X_test_patch_tensor), batch_size):
            batch_X = X_test_patch_tensor[i:i + batch_size]
            output = patchtst(batch_X).cpu().numpy()

# --- Fix for PatchTST multi-output shape ---
# If output is (batch, seq_len, 1) → take last timestep per batch
            if output.ndim == 3:
                output = output[:, -1, 0]
# If output is (batch, seq_len) → take last timestep
            elif output.ndim == 2:
                output = output[:, -1]
# If output is (batch, something larger), flatten safely
            elif output.ndim > 3:
                output = output.reshape(output.shape[0], -1)[:, -1]

            preds_patch.append(output)

    preds_patch = np.concatenate(preds_patch).flatten()
    print(f"✅ PatchTST Predictions: {len(preds_patch)} samples")

    # ===================================================================
    #  COMBINE RESULTS
    # ===================================================================
    if not (len(y_test) == len(preds_stock) == len(preds_patch)):
        raise ValueError(f"Length mismatch: y_test={len(y_test)}, Stockformer={len(preds_stock)}, PatchTST={len(preds_patch)}")

    results_df = pd.DataFrame({
        "timestamp": y_test_df.index,
        "Actual": y_test,
        "Stockformer_Pred": preds_stock,
        "PatchTST_Pred": preds_patch
    }).set_index("timestamp")

    results_df.to_csv(OUTPUT_PATH)
    print(f"\n✅ Saved combined predictions to {OUTPUT_PATH}")
    print(results_df.head())

    # ===================================================================
    #  EVALUATION METRICS
    # ===================================================================
    print("\n📊 Evaluation Metrics:\n")
    metrics_table = []

    for model in ["Stockformer_Pred", "PatchTST_Pred"]:
        mse = mean_squared_error(results_df["Actual"], results_df[model])
        mae = mean_absolute_error(results_df["Actual"], results_df[model])
        strategy_returns = results_df["Actual"] * results_df[model]
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-9)
        roi = (1 + strategy_returns).cumprod().iloc[-1] - 1

        metrics_table.append([model, mse, mae, roi * 100, sharpe])

    metrics_df = pd.DataFrame(metrics_table, columns=["Model", "MSE", "MAE", "ROI (%)", "Sharpe Ratio"])
    print(metrics_df.to_string(index=False, float_format="%.6f"))

    # ===================================================================
    #  VISUALIZATION
    # ===================================================================
    print("\n📈 Plotting results...")
    plt.figure(figsize=(10, 4))
    plt.plot(results_df.index, results_df["Actual"], label="Actual", alpha=0.7)
    plt.plot(results_df.index, results_df["Stockformer_Pred"], label="Stockformer", alpha=0.7)
    plt.plot(results_df.index, results_df["PatchTST_Pred"], label="PatchTST", alpha=0.7)
    plt.title("Predicted vs Actual Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for model in ["Stockformer_Pred", "PatchTST_Pred"]:
        strategy_returns = results_df["Actual"] * results_df[model]
        (1 + strategy_returns).cumprod().plot(label=f"{model} Strategy ROI")
    (1 + results_df["Actual"]).cumprod().plot(label="Buy & Hold ROI", linestyle="--")
    plt.title("Cumulative ROI Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n✅ Inference & evaluation complete.")


if __name__ == "__main__":
    run_inference()
