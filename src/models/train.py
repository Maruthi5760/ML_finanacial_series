import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import joblib
import json

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
# --- End Path Setup ---

# Import models and losses from our model.py file
from src.models.model import Stockformer, AsymmetricMSELoss, SharpeRatioLoss

# Import the SOTA model (Contribution 2)
from tsai.models.PatchTST import PatchTST

# --- 1. Helper Function: Create Time-Series Sequences ---
def create_sequences(X_data, y_data, seq_length):
    """
    Converts 2D feature/target data into 3D time-series sequences.
    """
    # Initialize empty lists
    X_seq, y_seq =[],[]
    
    for i in range(len(X_data) - seq_length + 1):
        X_seq.append(X_data[i:(i + seq_length)])
        y_seq.append(y_data[i + seq_length - 1])
        
    return np.array(X_seq), np.array(y_seq)

# --- 2. Helper Function: Main Training & Evaluation Loop ---
def run_training(model_name, model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, num_epochs, model_save_path):
    """
    Orchestrates the training and validation loops for a given model.
    Saves the best performing model.
    """
    print(f"\n--- Starting Training for {model_name} ---")
    
    best_val_loss = float('inf')
    # Initialize empty lists for history
    loss_history = {'train':[], 'val':[]}

    for epoch in range(num_epochs):
        # --- Training Step ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        loss_history['train'].append(avg_train_loss)

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        loss_history['val'].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Update learning rate scheduler
        if scheduler:
            scheduler.step(avg_val_loss)
            
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")

    print(f"--- Finished Training {model_name} ---")
    return loss_history

# --- 3. Main Orchestration Function ---
def main():
    """
    Main script to load, process, train, and save all components.
    """
    print("Starting the main training pipeline...")
    
    # --- Hyperparameters ---
    SEQ_LENGTH = 60       # 60-hour lookback window [3]
    TEST_SET_SIZE = 0.2
    VAL_SET_SIZE = 0.15   # 0.15 of the (1 - TEST_SET_SIZE) training data
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 5       # Set to 50 for a real run, 5-10 for a quick test [1]

    # --- Paths ---
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    
    FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
    TARGET_PATH = os.path.join(DATA_DIR, "target.csv")
    
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
    STOCKFORMER_PATH = os.path.join(MODEL_DIR, "stockformer_best.pth")
    PATCHTST_PATH = os.path.join(MODEL_DIR, "patchtst_best.pth")
    
    TEST_FEATURES_PATH_NPY = os.path.join(DATA_DIR, "X_test.npy")
    TEST_TARGET_PATH_CSV = os.path.join(DATA_DIR, "y_test.csv")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    X_df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    y_df = pd.read_csv(TARGET_PATH, index_col=0, parse_dates=True)
    
    # Store test timestamps for later
    y_timestamps = y_df.index
    
    # Convert to numpy arrays
    X_data = X_df.values
    y_data = y_df.values.squeeze() # Squeeze to 1D array
    
    print(f"Loaded raw data: X shape {X_data.shape}, y shape {y_data.shape}")

       # --- 2. Create Sequences ---
    X_seq, y_seq = create_sequences(X_data, y_data, SEQ_LENGTH)

    # --- FIX: Align timestamps properly ---
    # Each y_seq[i] corresponds to timestamp[i + SEQ_LENGTH - 1]
    y_seq_timestamps = y_timestamps[SEQ_LENGTH - 1:]  # slice off the first (SEQ_LENGTH-1)

    print(f"Created sequences: X shape {X_seq.shape}, y shape {y_seq.shape}")
    print(f"Aligned timestamps shape: {y_seq_timestamps.shape}")

    # Sanity check
    assert len(X_seq) == len(y_seq) == len(y_seq_timestamps), \
        f"Mismatch: X_seq={len(X_seq)}, y_seq={len(y_seq)}, ts={len(y_seq_timestamps)}"

    # --- 3. Train-Val-Test Split ---
    X_train_val, X_test, y_train_val, y_test, ts_train_val, ts_test = train_test_split(
        X_seq, y_seq, y_seq_timestamps, test_size=TEST_SET_SIZE, shuffle=False
    )

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_SET_SIZE, shuffle=False
    )
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")

    # --- 4. Scale Data ---
    # Reshape for scaler: (num_samples * seq_len, num_features)
    # --- BUG FIX 2: num_features is the 3rd dimension (index 2) ---
    num_features = X_train.shape[2]
    X_train_2d = X_train.reshape(-1, num_features)
    
    scaler = StandardScaler()
    scaler.fit(X_train_2d)
    
    # Save the scaler for predict.py
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")
    
    # Transform all sets
    X_train = scaler.transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    # --- 5. Save Test Data for predict.py ---
    np.save(TEST_FEATURES_PATH_NPY, X_test)
    y_test_df_save = pd.DataFrame(y_test, index=ts_test, columns=['target_return'])
    y_test_df_save.to_csv(TEST_TARGET_PATH_CSV)
    print(f"Saved test data to {TEST_FEATURES_PATH_NPY} and {TEST_TARGET_PATH_CSV}")
    
    # --- 6. Create PyTorch DataLoaders ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 7. Run Training for Baseline: Stockformer (Contribution 3 Loss) ---
    stockformer_model = Stockformer(num_features=num_features).to(device)
    loss_fn = AsymmetricMSELoss(alpha=2.5).to(device) # Contribution 3 [4]
    optimizer = optim.Adam(stockformer_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    sf_history = run_training(
        "Stockformer (AdjMSE)", stockformer_model, train_loader, val_loader, 
        loss_fn, optimizer, scheduler, device, NUM_EPOCHS, STOCKFORMER_PATH
    )

    # --- 8. Run Training for SOTA: PatchTST (Contribution 2) ---
    # PatchTST requires (batch, features, seq_len) [5, 6]
    # Our data is (batch, seq_len, features). We must permute.
    X_train_patch = np.transpose(X_train, (0, 2, 1)) # (N, 12, 60)
    X_val_patch = np.transpose(X_val, (0, 2, 1))     # (N, 12, 60)
    
    train_dataset_patch = TensorDataset(torch.tensor(X_train_patch, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset_patch = TensorDataset(torch.tensor(X_val_patch, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader_patch = DataLoader(train_dataset_patch, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_patch = DataLoader(val_dataset_patch, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize PatchTST [7]
        # Initialize PatchTST [7]
    patchtst_model = PatchTST(
        c_in=num_features,    # number of input features (channels)
        c_out=1,              # output dimension
        seq_len=SEQ_LENGTH,   # input sequence length
        pred_dim=1,           # prediction dimension (instead of pred_len)
        patch_len=10,         # length of each patch
        stride=5,             # stride between patches
        dropout=0.05,         # slight dropout for regularization
        revin=True,           # RevIN normalization enabled
        res_attention=True    # residual attention connections
    ).to(device)


    loss_fn = AsymmetricMSELoss(alpha=2.5).to(device) # Use same loss for fair comparison [4]
    optimizer = optim.Adam(patchtst_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    patch_history = run_training(
        "PatchTST (AdjMSE)", patchtst_model, train_loader_patch, val_loader_patch,
        loss_fn, optimizer, scheduler, device, NUM_EPOCHS, PATCHTST_PATH
    )

    # --- 9. Save Loss Histories ---
    all_history = {
        "stockformer": sf_history,
        "patchtst": patch_history
    }
    HISTORY_PATH = os.path.join(REPORTS_DIR, "loss_history.json")
    with open(HISTORY_PATH, 'w') as f:
        json.dump(all_history, f, indent=4)
    print(f"Saved all loss histories to {HISTORY_PATH}")
    print("\nTraining pipeline complete.")

if __name__ == "__main__":
    main()