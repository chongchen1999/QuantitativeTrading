import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
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
        self.scalers = {}  # Store scalers for each stock
        
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stock_data = []
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Create a date range for all possible trading days
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        for file in csv_files:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            
            # Calculate log returns
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Create a template DataFrame with all dates
            template = pd.DataFrame(index=all_dates)
            template.index.name = 'timestamp'
            
            # Merge with actual data and forward fill missing values
            df.set_index('timestamp', inplace=True)
            df = template.join(df)
            df['returns'] = df['returns'].ffill()
            
            # Reset index to get timestamp as column
            df.reset_index(inplace=True)
            
            # Normalize returns
            scaler = StandardScaler()
            df['returns'] = scaler.fit_transform(df[['returns']])
            self.scalers[file] = scaler
            
            # Drop any remaining NaN values (should only be the first day's return)
            df = df.dropna()
            
            stock_data.append(df)
        
        # Get common dates across all stocks
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
        
        # Create sequences
        for t in range(len(common_dates) - seq_len - 1):
            current_dates = common_dates[t:t+seq_len]
            next_date = common_dates[t+seq_len]
            
            X = np.zeros((len(stock_data), seq_len, 1))  # Only using returns
            y = np.zeros(len(stock_data))
            
            for i, df in enumerate(stock_data):
                window_data = df[df['timestamp'].isin(current_dates)]['returns'].values
                if len(window_data) == seq_len:
                    X[i, :, 0] = window_data
                    next_return = df[df['timestamp'] == next_date]['returns'].values
                    if len(next_return) > 0:
                        y[i] = next_return[0]
            
            self.data.append(X)
            self.labels.append(y)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.labels[idx])

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.output_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size, seq_length, d_model)
        x = self.output_layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))
        return x

class StockTransformer(nn.Module):
    def __init__(self, seq_len, num_stocks, num_features=1, d_model=128, num_heads=4, num_layers=4, dropout=0.1, early_stop=25):
        super().__init__()
        self.feature_embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_stocks, seq_len, num_features]
        batch_size, num_stocks, seq_len, num_features = x.shape
        
        # Reshape and embed features
        x = x.view(-1, seq_len, num_features)
        x = self.feature_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape and predict
        x = x.reshape(batch_size, num_stocks, -1)
        return self.fc(x).squeeze(-1)

def train_model(model, train_loader, val_loader, epochs=1000, lr=0.001, early_stop=25, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    warmup_epochs = 100  # 10% of total epochs for warmup
    counter = 0
    
    print(f"Training on {device}")
    print(f"Initial learning rate: {lr:.6e}")
    
    for epoch in range(epochs):
        # Learning rate warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate after warmup
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6e}')
            
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Stop if learning rate becomes too small
        if current_lr < 1e-6:
            print(f"Learning rate too small ({current_lr:.6e}), stopping training")
            break