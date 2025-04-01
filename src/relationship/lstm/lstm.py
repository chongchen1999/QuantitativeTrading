import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

class StockDataset(Dataset):
    def __init__(self, stock_data, sequence_length=10):
        """
        初始化数据集
        
        参数:
        stock_data: 形状为 [n_days, n_stocks] 的numpy数组，包含所有股票的对数收益率
        sequence_length: 序列长度，即使用多少天的历史数据来预测未来
        """
        self.sequence_length = sequence_length
        self.n_stocks = stock_data.shape[1]
        
        # 创建输入序列(X)和目标值(y)
        X, y = [], []
        for i in range(len(stock_data) - sequence_length - 1):
            # 输入: t 到 t+sequence_length-1 天的所有股票数据
            X.append(stock_data[i:i+sequence_length])
            # 输出: t+sequence_length+1 天的所有股票数据（下一天）
            y.append(stock_data[i+sequence_length+1])
        
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        初始化LSTM模型
        
        参数:
        input_size: 输入特征数量，等于股票数量
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        output_size: 输出特征数量，等于股票数量
        dropout: Dropout比率，用于防止过拟合
        """
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层：从LSTM的输出映射到每只股票的预测
        self.fc = nn.Linear(hidden_size, output_size)
        
        # 注意力矩阵：用于捕捉股票间的影响关系
        self.attention_matrix = nn.Parameter(torch.zeros(input_size, output_size))
        nn.init.xavier_uniform_(self.attention_matrix)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        x: 输入数据，形状为 [batch_size, sequence_length, n_stocks]
        
        返回:
        out: 模型输出，形状为 [batch_size, n_stocks]
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播，输出所有时间步的隐藏状态
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        return out
    
    def get_relation_matrix(self):
        """
        获取股票间的关系矩阵
        
        返回:
        relation_matrix: 形状为 [n_stocks, n_stocks] 的numpy数组
        """
        # 注意力矩阵经过 tanh 函数映射到 [-1, 1] 范围
        relation_matrix = torch.tanh(self.attention_matrix)
        return relation_matrix.detach().cpu().numpy()

def preprocess_data(stock_prices, window_size=1):
    """
    预处理股票价格数据
    
    参数:
    stock_prices: 形状为 [n_days, n_stocks] 的numpy数组，包含所有股票的收盘价
    window_size: 计算对数收益率的窗口大小，默认为1（日收益率）
    
    返回:
    log_returns: 形状为 [n_days-window_size, n_stocks] 的numpy数组，包含所有股票的对数收益率
    """
    log_returns = np.log(stock_prices[window_size:] / stock_prices[:-window_size])
    
    # 标准化对数收益率
    scaler = MinMaxScaler(feature_range=(-1, 1))
    log_returns_scaled = scaler.fit_transform(log_returns)
    
    return log_returns_scaled

def train_model(model, dataloader, num_epochs=100, learning_rate=0.001):
    """
    训练模型
    
    参数:
    model: LSTM模型
    dataloader: 数据加载器
    num_epochs: 训练轮数
    learning_rate: 学习率
    
    返回:
    model: 训练好的模型
    losses: 训练过程中的损失值列表
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return model, losses

def create_relation_heatmap(relation_matrix, stock_names=None):
    """
    创建股票关系矩阵的热力图
    
    参数:
    relation_matrix: 形状为 [n_stocks, n_stocks] 的numpy数组
    stock_names: 股票名称列表，默认为None（使用索引作为股票名称）
    """
    n_stocks = relation_matrix.shape[0]
    
    if stock_names is None:
        stock_names = [f'Stock {i+1}' for i in range(n_stocks)]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(relation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    
    # 设置坐标轴刻度和标签
    plt.xticks(np.arange(n_stocks), stock_names, rotation=45)
    plt.yticks(np.arange(n_stocks), stock_names)
    
    # 在每个格子上显示数值
    for i in range(n_stocks):
        for j in range(n_stocks):
            plt.text(j, i, f'{relation_matrix[i, j]:.2f}',
                     ha='center', va='center', 
                     color='white' if abs(relation_matrix[i, j]) > 0.5 else 'black')
    
    plt.xlabel('Stock (Effect)')
    plt.ylabel('Stock (Cause)')
    plt.title('Stock Relationship Matrix')
    plt.tight_layout()
    
    return plt

def main(stock_prices, stock_names=None, sequence_length=10, batch_size=32, 
         hidden_size=64, num_layers=2, num_epochs=100):
    """
    主函数，处理数据并训练模型
    
    参数:
    stock_prices: 形状为 [n_days, n_stocks] 的numpy数组，包含所有股票的收盘价
    stock_names: 股票名称列表，默认为None（使用索引作为股票名称）
    sequence_length: 序列长度，即使用多少天的历史数据来预测未来
    batch_size: 批处理大小
    hidden_size: LSTM隐藏层大小
    num_layers: LSTM层数
    num_epochs: 训练轮数
    
    返回:
    relation_matrix: 形状为 [n_stocks, n_stocks] 的numpy数组，表示股票间的关系
    """
    # 预处理数据
    log_returns = preprocess_data(stock_prices)
    
    # 创建数据集和数据加载器
    dataset = StockDataset(log_returns, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    n_stocks = stock_prices.shape[1]
    model = StockLSTM(n_stocks, hidden_size, num_layers, n_stocks)
    
    # 训练模型
    model, losses = train_model(model, dataloader, num_epochs=num_epochs)
    
    # 获取关系矩阵
    relation_matrix = model.get_relation_matrix()
    
    # 创建热力图
    plt = create_relation_heatmap(relation_matrix, stock_names)
    plt.savefig('stock_relation_matrix.png')
    
    print("训练完成，关系矩阵已保存为热力图")
    
    return relation_matrix

# 示例用法
if __name__ == "__main__":
    # 生成一些示例数据 (5支股票，100天)
    np.random.seed(42)
    n_days, n_stocks = 100, 5
    
    # 生成随机的股票价格数据
    initial_prices = np.random.rand(n_stocks) * 100 + 50  # 初始价格在50-150之间
    daily_returns = np.random.normal(1.001, 0.02, (n_days, n_stocks))  # 每日收益率均值为0.1%，标准差为2%
    
    # 计算每日价格
    stock_prices = np.zeros((n_days, n_stocks))
    stock_prices[0] = initial_prices
    for i in range(1, n_days):
        stock_prices[i] = stock_prices[i-1] * daily_returns[i]
    
    stock_names = [f'Stock {i+1}' for i in range(n_stocks)]
    
    # 运行主函数
    relation_matrix = main(stock_prices, stock_names, sequence_length=10, num_epochs=50)
    
    print("股票关系矩阵:")
    print(relation_matrix)