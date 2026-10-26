import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np

def build_features(ticker="ICICIBANK.NS", period="730d", interval="1h", timeout=30):
    """
    Downloads data for a given ticker and engineers a rich feature set.
    Fix: Added multi_level_index=False to yf.download to ensure a flat DataFrame.
    """
    # 1. Download raw data (as in baseline)
    # --- THIS IS THE NEW FIX ---
    # Explicitly set multi_level_index=False to prevent the MultiIndex error.
    df = yf.download(tickers=ticker, period=period, interval=interval, 
                     timeout=timeout, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # 2. Baseline Feature: Intra-hour Log Return
    df['log_return'] = np.log(df['Close'] / df['Open'])
    
    # Define our base features (what we start with)
    base_features = list(df.columns)

    # 3. Apply indicators one by one
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.sma(close='Volume', length=20, prefix='VOL', append=True)
    except Exception as e:
        print(f"Error applying technical indicators: {e}")
        raise

    # 5. Data Cleaning
    df = df.dropna()

    # 6. Define X (features) and y (target)
    df['target'] = df['log_return'].shift(-1)
    df = df.dropna()

    # Define feature columns dynamically
    indicator_cols = [col for col in df.columns if col not in base_features]
    feature_cols = base_features + indicator_cols
    
    feature_cols.remove('target')
    if 'Adj Close' in feature_cols:
        feature_cols.remove('Adj Close')

    X = df[feature_cols]
    
    # --- THIS IS THE SECOND FIX ---
    # Corrected NameError: from df[target] to df['target']
    y = df['target']
    
    return X, y, df

if __name__ == '__main__':
    try:
        # Pass the parameter to the test call as well
        X, y, df = build_features()
        print("Feature engineering successful.")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print("\nFeatures (X) head:")
        print(X.head())
    except Exception as e:
        print(f"An error occurred: {e}")