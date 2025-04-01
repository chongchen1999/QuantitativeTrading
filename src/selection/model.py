import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import glob

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
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
        
        # Read and process stock data
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stock_data = []
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        for file in csv_files:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            df = df.sort_values('timestamp')
            
            # Calculate returns and additional features
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['price_ma5'] = df['close'].rolling(window=5).mean()
            df['price_ma20'] = df['close'].rolling(window=20).mean()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            df = df.dropna()
            
            # Normalize features
            features = ['returns', 'volume_ma5', 'price_ma5', 'price_ma20', 'volatility']
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])
            
            stock_name = file.split('/')[-1].split('.')[0]
            self.scalers[stock_name] = scaler
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
            
            X = np.zeros((len(stock_data), seq_len, len(features)))
            y = np.zeros(len(stock_data))
            
            for i, df in enumerate(stock_data):
                window_data = df[df['timestamp'].isin(current_dates)][features].values
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
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.reshape(batch_size, seq_len, d_model)
        output = self.output_layer(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class EnhancedStockTransformer(nn.Module):
    def __init__(self, seq_len, num_stocks, num_features=5, d_model=128, num_heads=8, 
                 num_layers=6, d_ff=512, dropout=0.1):
        super().__init__()
        
        self.feature_embedding = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # x shape: [batch_size, num_stocks, seq_len, num_features]
        batch_size, num_stocks, seq_len, num_features = x.shape
        
        # Process each stock separately
        outputs = []
        for i in range(num_stocks):
            stock_data = x[:, i]  # [batch_size, seq_len, num_features]
            
            # Embed features
            embedded = self.feature_embedding(stock_data)
            
            # Add positional encoding
            encoded = self.positional_encoding(embedded)
            encoded = self.dropout(encoded)
            
            # Apply transformer blocks
            for transformer_block in self.transformer_blocks:
                encoded = transformer_block(encoded)
            
            # Reshape and predict
            flattened = encoded.reshape(batch_size, -1)
            output = self.output_layer(flattened)
            outputs.append(output)
        
        # Combine predictions for all stocks
        return torch.cat(outputs, dim=1)  # [batch_size, num_stocks]