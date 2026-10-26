import torch
import torch.nn as nn
import math

# --- 1. Positional Encoding (Batch First) ---
# This is a standard Transformer utility class, modified to accept
# (batch_size, seq_len, d_model) inputs, which is common.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- 2. Baseline "Stockformer" Model ---
# This is the model from your TR_1.0.ipynb, implemented as a
# sequence-to-one (many-to-one) predictor. [3, 4]
class Stockformer(nn.Module):
    def __init__(self, num_features, embedding_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        super(Stockformer, self).__init__()
        self.model_type = 'Transformer'
        
        # 1D-CNN Feature Extractor [3]
        # Takes (batch, num_features, seq_len)
        self.embedding = nn.Conv1d(in_channels=num_features, out_channels=embedding_dim,
                                   kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Positional Encoding (batch_first)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        # Standard Transformer Encoder (batch_first) [4]
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output Layer
        # Takes the final hidden state of the last token
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, src):
        """
        Input src shape: (batch_size, seq_len, num_features)
        """
        
        # 1. Permute for 1D-CNN: (batch_size, num_features, seq_len)
        src = src.permute(0, 2, 1)
        
        # 2. Apply CNN Embedding: (batch_size, embedding_dim, seq_len)
        src = self.relu(self.embedding(src))
        
        # 3. Permute back for Transformer: (batch_size, seq_len, embedding_dim)
        src = src.permute(0, 2, 1)
        
        # 4. Apply Positional Encoding
        src = self.pos_encoder(src)
        
        # 5. Apply Transformer Encoder
        # Output shape: (batch_size, seq_len, embedding_dim)
        transformer_output = self.transformer_encoder(src)
        
        # 6. Get output for the *last* time step for prediction
        # Shape: (batch_size, embedding_dim)
        output = transformer_output[:, -1, :] 
        
        # 7. Apply Final Linear Layer
        # Shape: (batch_size, 1)
        output = self.output_layer(output)
        
        # Return a flat tensor: (batch_size)
        return output.squeeze(-1)

# --- 3. Custom Loss Functions (Contribution 3) ---

class AsymmetricMSELoss(nn.Module):
    """
    Implements the AdjMSE2 loss from research [1], which
    penalizes predictions with the wrong sign more heavily.
    """
    def __init__(self, alpha=2.5):
        super().__init__()
        self.alpha = alpha
        # We use 'none' reduction to get per-element losses
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        # Calculate standard MSE for all elements
        loss_elements = self.mse(y_pred, y_true)
        
        # Create a mask for predictions in the wrong direction
        # (y_pred * y_true < 0)
        wrong_direction_mask = (y_pred * y_true) < 0
        
        # Apply the alpha penalty to those elements
        loss_elements[wrong_direction_mask] *= self.alpha
        
        # Return the mean of all (potentially penalized) losses
        return torch.mean(loss_elements)

class SharpeRatioLoss(nn.Module):
    """
    Directly optimizes the (negative) Sharpe Ratio.[5]
    Assumes y_pred is the unbounded signal/position size and 
    y_true is the actual return for the period.
    This logic follows your mid-term report's goal. [3]
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # 1. Calculate portfolio returns
        # (signal * actual_return)
        portfolio_returns = y_pred * y_true
        
        # 2. Calculate Sharpe Ratio components
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns)
        
        # 3. Calculate Sharpe Ratio (add epsilon for numerical stability [6])
        sharpe_ratio = mean_return / (std_return + 1e-9)
        
        # 4. Return the negative Sharpe Ratio (for minimization)
        return -sharpe_ratio