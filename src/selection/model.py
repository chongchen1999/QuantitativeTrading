import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockDataset(Dataset):
    def __init__(self, data_dir, start_date, end_date, seq_len):
        self.seq_len = seq_len
        self.data = []
        self.labels = []
        
        # Read all stock data
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stock_data = []
        
        # Ensure consistent date format
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        for file in csv_files:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            df = df.sort_values('timestamp')
            
            # Calculate log returns
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            df = df.dropna()
            stock_data.append(df)
        
        # Find common trading days
        common_dates = None
        for df in stock_data:
            dates = set(df['timestamp'])
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)
        
        common_dates = sorted(list(common_dates))
        print(f"Found {len(common_dates)} common trading days")
        
        if len(common_dates) < seq_len + 1:
            raise ValueError(f"Not enough common trading days ({len(common_dates)}) for sequence length {seq_len}")
        
        # Build training data
        for t in range(len(common_dates) - seq_len - 1):
            current_dates = common_dates[t:t+seq_len]
            next_date = common_dates[t+seq_len]
            
            X = np.zeros((len(stock_data), seq_len))
            y = np.zeros(len(stock_data))
            
            for i, df in enumerate(stock_data):
                window_data = df[df['timestamp'].isin(current_dates)]['returns'].values
                if len(window_data) == seq_len:
                    X[i] = window_data
                    next_return = df[df['timestamp'] == next_date]['returns'].values
                    if len(next_return) > 0:
                        y[i] = next_return[0]
            
            self.data.append(X)
            self.labels.append(y)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.labels[idx])

class EnhancedStockTransformer(nn.Module):
    def __init__(self, seq_len, num_stocks, d_model=64, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        encoder_layers = []
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            encoder_layers.append(layer)
        self.transformer_layers = nn.ModuleList(encoder_layers)
        
        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model * 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, mask=None):
        batch_size, num_stocks, seq_len = x.shape
        
        x = x.view(-1, seq_len, 1)
        x = self.feature_embedding(x)
        x = self.pos_encoder(x)
        
        skip_x = x
        for transformer_layer, skip_layer in zip(self.transformer_layers, self.skip_connections):
            x = transformer_layer(x, src_key_padding_mask=mask if mask is not None else None)
            x = x + skip_layer(skip_x)
            
        x = x.reshape(batch_size * num_stocks, -1)
        x = self.output_head(x)
        x = x.view(batch_size, num_stocks)
        
        return x

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"Training on {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss/len(train_loader):.6f}, '
                  f'Val Loss: {val_loss:.6f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def plot_predictions(model, test_loader, stock_names, device='cuda'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            predictions.append(pred.cpu().numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Individual stock comparison plots
    n_stocks = len(stock_names)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(15, 5*n_stocks))
    for i, (ax, stock_name) in enumerate(zip(axes, stock_names)):
        ax.plot(predictions[:, i], label='Predicted', color='blue', alpha=0.7)
        ax.plot(actuals[:, i], label='Actual', color='red', alpha=0.7)
        ax.set_title(f'{stock_name} Return Rate Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel('Return Rate')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('individual_stock_comparisons.png')
    plt.close()
    
    # All stocks actual returns comparison
    plt.figure(figsize=(15, 8))
    for i, stock_name in enumerate(stock_names):
        plt.plot(actuals[:, i], label=stock_name)
    plt.title('Actual Return Rates Comparison')
    plt.xlabel('Time')
    plt.ylabel('Return Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_returns_comparison.png')
    plt.close()
    
    # All stocks predicted returns comparison
    plt.figure(figsize=(15, 8))
    for i, stock_name in enumerate(stock_names):
        plt.plot(predictions[:, i], label=stock_name)
    plt.title('Predicted Return Rates Comparison')
    plt.xlabel('Time')
    plt.ylabel('Return Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predicted_returns_comparison.png')
    plt.close()
    
    # Print performance metrics
    print("\nPrediction Performance Metrics:")
    for i, stock_name in enumerate(stock_names):
        mse = np.mean((predictions[:, i] - actuals[:, i])**2)
        mae = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        correlation = np.corrcoef(predictions[:, i], actuals[:, i])[0, 1]
        print(f"\n{stock_name}:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Correlation: {correlation:.6f}")