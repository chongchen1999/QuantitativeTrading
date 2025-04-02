# model.py
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
    def __init__(self, data_dir, start_date, end_date, seq_len, fill_method='forward', min_valid_ratio=0.8):
        """
        Initialize the stock dataset.
        
        Args:
            data_dir: Directory containing stock CSV files
            start_date: Start date for the dataset
            end_date: End date for the dataset
            seq_len: Length of each sequence
            fill_method: Method to fill missing values ('forward', 'mean', or 'zero')
            min_valid_ratio: Minimum ratio of valid data points required in a sequence
        """
        self.seq_len = seq_len
        self.fill_method = fill_method
        self.min_valid_ratio = min_valid_ratio
        self.data = []
        self.labels = []
        self.stock_indices = []  # Keep track of which stocks are in each sequence
        self.scalers = {}  # Store scalers for each stock
        
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stock_data = []
        stock_files = []  # Keep track of which files were successfully processed
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Process each stock file
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                
                # Skip if dataframe is empty after date filtering
                if len(df) == 0:
                    print(f"Skipping {file} - no data within date range")
                    continue
                    
                df = df.sort_values('timestamp')
                
                # Calculate multiple features
                df['returns'] = np.log(df['close'] / df['close'].shift(1))
                df['volume_ma5'] = df['volume'].rolling(window=5).mean()
                df['price_ma5'] = df['close'].rolling(window=5).mean()
                df['price_ma20'] = df['close'].rolling(window=20).mean()
                df['volatility'] = df['returns'].rolling(window=20).std()
                
                # Create feature matrix
                features = ['returns', 'volume_ma5', 'price_ma5', 'price_ma20', 'volatility']
                
                # Handle missing values based on the specified method
                self._handle_missing_values(df, features)
                
                # Skip if the dataframe has no valid rows after handling missing values
                if len(df.dropna(subset=features)) == 0:
                    print(f"Skipping {file} - no valid data after handling missing values")
                    continue
                
                # Normalize features - use only non-NaN values for fitting the scaler
                valid_data = df[features].dropna()
                if len(valid_data) == 0:
                    print(f"Skipping {file} - no valid data for scaling")
                    continue
                    
                scaler = StandardScaler()
                scaler.fit(valid_data)  # Fit on valid data only
                
                # Transform the entire dataset, NaNs will remain NaNs
                df[features] = pd.DataFrame(
                    scaler.transform(df[features].fillna(0)),  # Temporarily fill NaNs for transform
                    columns=features,
                    index=df.index
                )
                
                # Replace NaNs back if there were any after scaling
                for feature in features:
                    df[feature] = np.where(df[feature].isna(), np.nan, df[feature])
                
                # Store the scaler
                self.scalers[file] = scaler
                
                # Keep valid rows (rows with all features valid)
                df = df.dropna(subset=features)
                
                # Check if there's still enough data after dropping NaNs
                if len(df) <= self.seq_len:
                    print(f"Skipping {file} - not enough data points after dropping NaNs")
                    continue
                
                stock_data.append(df)
                stock_files.append(file)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        if len(stock_data) == 0:
            raise ValueError("No valid stock data found after preprocessing")
            
        print(f"Successfully processed {len(stock_data)} out of {len(csv_files)} stocks")
        
        # Find common trading dates across all stocks
        self._process_common_dates(stock_data, stock_files, features)
    
    def _handle_missing_values(self, df, features):
        """
        Handle missing values in the dataframe.
        
        Args:
            df: DataFrame containing stock data
            features: List of feature columns
        """
        # Count missing values
        missing_count = df[features].isna().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values in data")
            
            if self.fill_method == 'forward':
                # Forward fill followed by backward fill to handle any remaining NaNs
                df[features] = df[features].ffill().bfill()
            elif self.fill_method == 'mean':
                # Fill with column means
                for feature in features:
                    mean_value = df[feature].mean()
                    df[feature] = df[feature].fillna(mean_value)
            elif self.fill_method == 'zero':
                # Fill with zeros
                df[features] = df[features].fillna(0)
            else:
                raise ValueError(f"Unsupported fill method: {self.fill_method}")
    
    def _process_common_dates(self, stock_data, stock_files, features):
        """
        Process common trading dates and create sequences.
        
        Args:
            stock_data: List of DataFrames containing processed stock data
            stock_files: List of file paths corresponding to stock_data
            features: List of feature columns
        """
        if not stock_data:
            raise ValueError("No stock data available for processing")
        
        # Find all unique dates across all stocks
        all_dates = set()
        stocks_by_date = {}
        
        for i, df in enumerate(stock_data):
            dates = set(df['timestamp'])
            all_dates.update(dates)
            
            # Map each date to the stocks that have data for it
            for date in dates:
                if date not in stocks_by_date:
                    stocks_by_date[date] = []
                stocks_by_date[date].append(i)
        
        all_dates = sorted(list(all_dates))
        print(f"Found {len(all_dates)} unique trading days across all stocks")
        
        # Filter dates that have data for at least min_valid_ratio of stocks
        min_stocks_required = int(len(stock_data) * self.min_valid_ratio)
        valid_dates = [date for date in all_dates if len(stocks_by_date[date]) >= min_stocks_required]
        valid_dates.sort()
        
        print(f"Found {len(valid_dates)} dates with at least {min_stocks_required} stocks having data")
        
        if len(valid_dates) < self.seq_len + 1:
            raise ValueError(f"Not enough valid trading days ({len(valid_dates)}) for sequence length {self.seq_len}")
        
        # Create sequences
        valid_sequences = 0
        skipped_sequences = 0
        
        # Find the minimum number of common stocks across all sequence windows
        # to ensure consistent tensor dimensions
        min_common_stocks = float('inf')
        sequence_stocks = []
        
        # First pass: find common stocks for each sequence window
        for t in range(len(valid_dates) - self.seq_len):
            if t + self.seq_len >= len(valid_dates):
                continue
                
            current_dates = valid_dates[t:t+self.seq_len]
            next_date = valid_dates[t+self.seq_len]
            
            # Find all stocks that have data for these dates
            stocks_in_window = set()
            for date in current_dates:
                if not stocks_in_window:
                    stocks_in_window = set(stocks_by_date[date])
                else:
                    stocks_in_window &= set(stocks_by_date[date])
            
            # Also need stocks that have the label date
            stocks_with_label = set(stocks_by_date[next_date])
            valid_stocks = list(stocks_in_window & stocks_with_label)
            
            # Skip if not enough stocks for this window
            if len(valid_stocks) < min_stocks_required:
                skipped_sequences += 1
                continue
            
            sequence_stocks.append(valid_stocks)
            min_common_stocks = min(min_common_stocks, len(valid_stocks))
        
        print(f"Minimum common stocks across all sequences: {min_common_stocks}")
        
        # Second pass: create sequences with the same number of stocks
        for t, valid_stocks in enumerate(sequence_stocks):
            current_dates = valid_dates[t:t+self.seq_len]
            next_date = valid_dates[t+self.seq_len]
            
            # Take the first min_common_stocks stocks to ensure consistent dimensions
            valid_stocks = valid_stocks[:min_common_stocks]
            
            # Create feature tensor and label vector for this window
            X = np.zeros((len(valid_stocks), self.seq_len, len(features)))
            y = np.zeros(len(valid_stocks))
            
            for i, stock_idx in enumerate(valid_stocks):
                df = stock_data[stock_idx]
                
                # Get data for the current window
                window_data = df[df['timestamp'].isin(current_dates)][features].values
                
                # Make sure we have complete data
                if len(window_data) == self.seq_len:
                    X[i] = window_data
                    
                    # Get label (next return)
                    next_return = df[df['timestamp'] == next_date]['returns'].values[0]
                    y[i] = next_return
            
            # Store the sequence
            self.data.append(X)
            self.labels.append(y)
            self.stock_indices.append(valid_stocks)  # Keep track of which stocks are in each sequence
            valid_sequences += 1
        
        print(f"Created {valid_sequences} valid sequences, skipped {skipped_sequences} sequences due to insufficient data")
        print(f"Each sequence contains data for {min_common_stocks} stocks")
        
        # Make sure we have at least some sequences
        if valid_sequences == 0:
            raise ValueError("No valid sequences could be created with the current parameters")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.labels[idx])

# class StockDataset(Dataset):
#     def __init__(self, data_dir, start_date, end_date, seq_len):
#         self.seq_len = seq_len
#         self.data = []
#         self.labels = []
#         self.scalers = {}  # Store scalers for each stock
        
#         csv_files = glob.glob(f"{data_dir}/*.csv")
#         stock_data = []
        
#         start_date = pd.to_datetime(start_date)
#         end_date = pd.to_datetime(end_date)
        
#         for file in csv_files:
#             df = pd.read_csv(file)
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
#             df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
#             df = df.sort_values('timestamp')
            
#             # Calculate multiple features
#             df['returns'] = np.log(df['close'] / df['close'].shift(1))
#             df['volume_ma5'] = df['volume'].rolling(window=5).mean()
#             df['price_ma5'] = df['close'].rolling(window=5).mean()
#             df['price_ma20'] = df['close'].rolling(window=20).mean()
#             df['volatility'] = df['returns'].rolling(window=20).std()
            
#             # Create feature matrix
#             features = ['returns', 'volume_ma5', 'price_ma5', 'price_ma20', 'volatility']
#             df = df.dropna()
            
#             # Normalize features
#             scaler = StandardScaler()
#             df[features] = scaler.fit_transform(df[features])
#             self.scalers[file] = scaler
            
#             stock_data.append(df)
        
#         common_dates = None
#         for df in stock_data:
#             dates = set(df['timestamp'])
#             if common_dates is None:
#                 common_dates = dates
#             else:
#                 common_dates = common_dates.intersection(dates)
        
#         common_dates = sorted(list(common_dates))
#         print(f"Found {len(common_dates)} common trading days")
        
#         if len(common_dates) < seq_len + 1:
#             raise ValueError(f"Not enough common trading days ({len(common_dates)}) for sequence length {seq_len}")
        
#         for t in range(len(common_dates) - seq_len - 1):
#             current_dates = common_dates[t:t+seq_len]
#             next_date = common_dates[t+seq_len]
            
#             X = np.zeros((len(stock_data), seq_len, len(features)))  # Modified for multiple features
#             y = np.zeros(len(stock_data))
            
#             for i, df in enumerate(stock_data):
#                 window_data = df[df['timestamp'].isin(current_dates)][features].values
#                 if len(window_data) == seq_len:
#                     X[i] = window_data
#                     next_return = df[df['timestamp'] == next_date]['returns'].values
#                     if len(next_return) > 0:
#                         y[i] = next_return[0]
            
#             self.data.append(X)
#             self.labels.append(y)
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.labels[idx])

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
    def __init__(self, seq_len, num_stocks, num_features=5, d_model=128, num_heads=8, num_layers=6, dropout=0.1):
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
    
def train_model(model, train_loader, val_loader, epochs=2000, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    warmup_epochs = 200  # 10% of total epochs for warmup
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
        
        if epoch % 2 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6e}')
            
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            counter = 0
        else:
            counter += 1
            if counter >= 50:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Stop if learning rate becomes too small
        if current_lr < 1e-6:
            print(f"Learning rate too small ({current_lr:.6e}), stopping training")
            break

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
    
    # 为每支股票创建单独的对比图
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
    
    # 创建所有股票实际收益率的对比图
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
    
    # 创建所有股票预测收益率的对比图
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
    
    # 计算并打印每支股票的预测准确度指标
    print("\nPrediction Performance Metrics:")
    for i, stock_name in enumerate(stock_names):
        mse = np.mean((predictions[:, i] - actuals[:, i])**2)
        mae = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
        correlation = np.corrcoef(predictions[:, i], actuals[:, i])[0, 1]
        print(f"\n{stock_name}:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Correlation: {correlation:.6f}")

    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    plt.figure(figsize=(15, 10))
    for i in range(len(stock_names)):
        plt.plot(predictions[:, i], label=f'{stock_names[i]} (pred)')
        plt.plot(actuals[:, i], label=f'{stock_names[i]} (actual)')
    
    plt.legend()
    plt.title('Predicted vs Actual Returns')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.show()