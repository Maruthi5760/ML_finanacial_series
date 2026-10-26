import pandas as pd
import os
import sys

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
# --- End Path Setup ---

from src.data.build_features import build_features

def main():
    """
    Runs the data processing pipeline:
    1. Downloads raw data and engineers features using build_features().
    2. Saves the processed features (X) and target (y) to 'data/processed/'.
    """
    print("Running data processing pipeline...")

    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "features.csv")
    TARGET_PATH = os.path.join(PROCESSED_DATA_DIR, "target.csv")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Ensured output directory exists: {PROCESSED_DATA_DIR}")

    # 1. Download and engineer features
    try:
        # --- THIS IS THE FIX ---
        # Removed the multi_level_index parameter from the call,
        # as the function now handles it internally.
        X, y, df_full = build_features(
            ticker="ICICIBANK.NS", 
            period="730d", 
            interval="1h",
            timeout=30
        )
        print("Feature engineering successful.")
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return

    # 2. Save the processed data
    print(f"Saving features to {FEATURES_PATH}...")
    X.to_csv(FEATURES_PATH, index=True)

    print(f"Saving target to {TARGET_PATH}...")
    y.to_csv(TARGET_PATH, index=True, header=['target'])

    print("\nData processing complete.")
    print(f"Features (X) saved with shape: {X.shape}")
    print(f"Target (y) saved with shape: {y.shape}")

if __name__ == "__main__":
    main()