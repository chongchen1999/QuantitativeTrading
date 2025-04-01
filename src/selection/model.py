import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import glob
import matplotlib.pyplot as plt

class StockDataset(Dataset):
    def __init__(self, data_dir, start_date, end_date, seq_len):
        self.seq_len = seq_len
        self.data = []
        self.labels = []
        
        # 读取所有股票数据
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stock_data = []
        
        # 确保所有日期格式一致
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        for file in csv_files:
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            df = df.sort_values('timestamp')  # 确保按时间排序
            
            # 计算log收益率
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            df = df.dropna()
            stock_data.append(df)
        
        # 找到所有股票共同的交易日
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
        
        # 构建训练数据
        for t in range(len(common_dates) - seq_len - 1):
            current_dates = common_dates[t:t+seq_len]
            next_date = common_dates[t+seq_len]
            
            X = np.zeros((len(stock_data), seq_len))
            y = np.zeros(len(stock_data))
            
            for i, df in enumerate(stock_data):
                # 获取当前时间窗口的数据
                window_data = df[df['timestamp'].isin(current_dates)]['returns'].values
                if len(window_data) == seq_len:  # 确保数据完整
                    X[i] = window_data
                    # 获取下一天的收益率
                    next_return = df[df['timestamp'] == next_date]['returns'].values
                    if len(next_return) > 0:
                        y[i] = next_return[0]
            
            self.data.append(X)
            self.labels.append(y)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.labels[idx])

class StockTransformer(nn.Module):
    def __init__(self, seq_len, num_stocks, d_model=64, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * seq_len, 1)
        
    def forward(self, x):
        # x shape: [batch_size, num_stocks, seq_len]
        batch_size, num_stocks, seq_len = x.shape
        
        # Reshape for embedding
        x = x.view(-1, seq_len, 1)
        x = self.embedding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Reshape and predict
        x = x.reshape(batch_size, num_stocks, -1)
        return self.fc(x).squeeze(-1)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
        
        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss/len(train_loader):.6f}, '
                  f'Val Loss: {val_loss/len(val_loader):.6f}')

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