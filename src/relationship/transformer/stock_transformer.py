import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from sklearn.preprocessing import MinMaxScaler
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("stock_transformer.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="股票关系分析Transformer模型")
    parser.add_argument("--data_dir", type=str, required=True, help="股票CSV文件所在目录")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期，格式：YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, required=True, help="结束日期，格式：YYYY-MM-DD")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--seq_len", type=int, default=10, help="序列长度")
    parser.add_argument("--d_model", type=int, default=64, help="模型维度")
    parser.add_argument("--nhead", type=int, default=8, help="多头注意力头数")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备")
    return parser.parse_args()

# 数据处理类
class StockDataProcessor:
    def __init__(self, data_dir, start_date, end_date):
        self.data_dir = data_dir
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.stock_data = {}
        self.tickers = []
        self.log_returns = {}
        self.processed_data = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """加载股票数据"""
        logger.info("加载股票数据...")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            ticker = file.split('.')[0]  # 假设文件名为 TICKER.csv
            self.tickers.append(ticker)
            
            # 读取CSV文件
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            
            # 确保时间戳列格式一致
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 筛选时间范围内的数据
            df = df[(df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)]
            
            # 按时间排序
            df = df.sort_values('timestamp')
            
            # 只保留需要的列
            df = df[['timestamp', 'close']]
            
            # 存储数据
            self.stock_data[ticker] = df
        
        logger.info(f"加载了 {len(self.tickers)} 支股票的数据")
        return self.stock_data
    
    def calculate_log_returns(self):
        """计算每支股票的log return"""
        logger.info("计算log returns...")
        for ticker, df in self.stock_data.items():
            # 计算log return
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            # 删除第一行（NaN值）
            df = df.dropna()
            self.log_returns[ticker] = df
        return self.log_returns
    
    def align_timestamps(self):
        """确保所有股票数据使用相同的时间戳"""
        logger.info("对齐时间戳...")
        # 获取所有时间戳
        all_timestamps = set()
        for df in self.log_returns.values():
            all_timestamps.update(df['timestamp'].tolist())
        
        # 按时间排序
        all_timestamps = sorted(list(all_timestamps))
        
        # 创建一个包含所有时间戳的DataFrame
        aligned_data = pd.DataFrame({'timestamp': all_timestamps})
        
        # 将每支股票的log_return添加到DataFrame中
        for ticker, df in self.log_returns.items():
            # 创建一个以timestamp为索引的Series
            ticker_series = pd.Series(df['log_return'].values, index=df['timestamp'])
            # 重新索引以包含所有时间戳，并填充缺失值
            aligned_data[ticker] = aligned_data['timestamp'].map(lambda x: ticker_series.get(x, np.nan))
        
        # 删除任何包含NaN的行
        aligned_data = aligned_data.dropna()
        
        self.processed_data = aligned_data
        return aligned_data
    
    def prepare_training_data(self, seq_len):
        """准备训练数据"""
        logger.info("准备训练数据...")
        # 提取features (不包括timestamp)
        features = self.processed_data.drop('timestamp', axis=1).values
        
        # 标准化特征
        normalized_features = self.scaler.fit_transform(features)
        
        # 创建序列数据
        X, y = [], []
        for i in range(len(normalized_features) - seq_len - 1):
            # 输入：seq_len天的数据
            X.append(normalized_features[i:i+seq_len])
            # 输出：下一天的数据
            y.append(normalized_features[i+seq_len+1])
        
        return np.array(X), np.array(y), self.processed_data.drop('timestamp', axis=1).columns.tolist()

# 数据集类
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 注意力掩码函数
def create_mask(seq_len, device):
    """创建上三角掩码，用于避免关注未来的token"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.to(device)

# 检查PyTorch版本的函数
def check_pytorch_version():
    """检查PyTorch版本并返回是否支持src_mask参数"""
    try:
        import torch
        version = torch.__version__
        logger.info(f"PyTorch版本: {version}")
        # 尝试确定是否支持src_mask
        # 此处只是一个简单的版本检查，可能需要更具体的逻辑
        major, minor = map(int, version.split('.')[:2])
        # 返回是否支持src_mask的猜测
        return major >= 1 and minor >= 2
    except Exception as e:
        logger.warning(f"无法检查PyTorch版本: {e}")
        return False

# Transformer模型
class StockTransformer(nn.Module):
    def __init__(self, n_stocks, seq_len, d_model, nhead, num_layers, dropout=0.1):
        super(StockTransformer, self).__init__()
        
        self.n_stocks = n_stocks
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 输入投影层（将每个股票的1维特征转换为d_model维）
        self.input_projection = nn.Linear(n_stocks, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出预测层
        self.output_layer = nn.Linear(d_model, n_stocks * n_stocks)
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, n_stocks]
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # Transformer编码器
        if mask is not None:
            # 尝试不同的掩码参数名称以兼容不同版本的PyTorch
            try:
                x = self.transformer_encoder(x, src_mask=mask)
            except TypeError:
                try:
                    x = self.transformer_encoder(x, mask=mask)
                except TypeError:
                    # 如果都不支持，不使用掩码
                    logger.warning("当前PyTorch版本不支持掩码参数，将不使用掩码")
                    x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x)
        
        # 使用最后一个时间步来预测关系矩阵
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # 输出层
        x = self.output_layer(x)  # [batch_size, n_stocks * n_stocks]
        
        # 重塑为关系矩阵
        x = x.view(batch_size, self.n_stocks, self.n_stocks)
        
        # 使用tanh确保输出在[-1, 1]范围内
        x = torch.tanh(x)
        
        return x

# 训练函数
def train_model(model, dataloader, optimizer, criterion, device, mask=None):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(X_batch, mask)
        
        # 计算关系矩阵（目标是下一天的log return）
        target_relations = calculate_target_relations(X_batch, y_batch)
        
        # 计算损失
        loss = criterion(outputs, target_relations)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 计算目标关系矩阵
def calculate_target_relations(X, y):
    """
    计算目标关系矩阵：
    - X: [batch_size, seq_len, n_stocks]
    - y: [batch_size, n_stocks]
    - 返回: [batch_size, n_stocks, n_stocks]，其中R[i][j]表示股票i对股票j第二天表现的影响
    """
    batch_size, seq_len, n_stocks = X.shape
    
    # 获取最后一天的数据
    last_day = X[:, -1, :]  # [batch_size, n_stocks]
    
    # 准备目标关系矩阵
    relations = torch.zeros(batch_size, n_stocks, n_stocks, device=X.device)
    
    # 使用更高效的向量化计算
    for b in range(batch_size):
        for i in range(n_stocks):
            for j in range(n_stocks):
                if i != j:  # 我们只关注不同股票之间的关系
                    # 股票i的当前log return
                    stock_i_current = last_day[b, i].item()  # 转换为Python标量以防止梯度跟踪
                    
                    # 股票j的下一天log return
                    stock_j_next = y[b, j].item()  # 转换为Python标量以防止梯度跟踪
                    
                    # 防止除以零
                    try:
                        # 简单的相关性计算（目标是-1到1之间的值）
                        if stock_i_current > 0 and stock_j_next > 0:
                            relations[b, i, j] = min(stock_i_current, stock_j_next) / max(stock_i_current, stock_j_next)
                        elif stock_i_current < 0 and stock_j_next < 0:
                            relations[b, i, j] = min(abs(stock_i_current), abs(stock_j_next)) / max(abs(stock_i_current), abs(stock_j_next))
                        elif stock_i_current > 0 and stock_j_next < 0:
                            relations[b, i, j] = -min(stock_i_current, abs(stock_j_next)) / max(stock_i_current, abs(stock_j_next))
                        elif stock_i_current < 0 and stock_j_next > 0:
                            relations[b, i, j] = -min(abs(stock_i_current), stock_j_next) / max(abs(stock_i_current), stock_j_next)
                        else:  # 如果任一值为0
                            relations[b, i, j] = 0
                    except (ZeroDivisionError, ValueError):
                        # 处理可能的除以零错误
                        relations[b, i, j] = 0
                        logger.warning(f"计算关系时出现错误：stock_i={stock_i_current}, stock_j={stock_j_next}")
    
    return relations

# 评估函数
def evaluate_model(model, dataloader, criterion, device, mask=None):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch, mask)
            
            # 计算目标关系矩阵
            target_relations = calculate_target_relations(X_batch, y_batch)
            
            # 计算损失
            loss = criterion(outputs, target_relations)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 可视化关系矩阵
def plot_relation_matrix(relation_matrix, tickers, output_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(relation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Stock Relation Matrix')
    plt.xlabel('Target Stock (Next Day)')
    plt.ylabel('Source Stock (Current Day)')
    
    # 添加刻度标签
    plt.xticks(np.arange(len(tickers)), tickers, rotation=45)
    plt.yticks(np.arange(len(tickers)), tickers)
    
    # 添加文本标注
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            plt.text(j, i, f'{relation_matrix[i, j]:.2f}',
                     ha='center', va='center', 
                     color='white' if abs(relation_matrix[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 主函数
def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置设备
        device = torch.device(args.device)
        logger.info(f"使用设备: {device}")
        
        # 检查PyTorch版本
        supports_src_mask = check_pytorch_version()
        
        # 数据处理
        processor = StockDataProcessor(args.data_dir, args.start_date, args.end_date)
        processor.load_data()
        processor.calculate_log_returns()
        processor.align_timestamps()
        X, y, tickers = processor.prepare_training_data(args.seq_len)
        
        logger.info(f"准备了 {X.shape[0]} 个训练样本，{len(tickers)} 支股票")
        
        # 检查训练样本数量
        if X.shape[0] <= 1:
            logger.error(f"训练样本数量太少: {X.shape[0]}")
            raise ValueError("训练样本数量不足，请增加时间范围或减小序列长度")
        
        # 划分训练集和验证集
        train_size = max(1, int(0.8 * len(X)))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 确保验证集不为空
        if len(X_val) == 0:
            logger.warning("验证集为空，使用训练集的一部分作为验证集")
            X_val, y_val = X_train, y_train
        
        # 创建数据加载器
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        
        # 调整批次大小以避免批次为空
        batch_size = min(args.batch_size, len(train_dataset))
        if batch_size != args.batch_size:
            logger.warning(f"调整批次大小: {args.batch_size} -> {batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 创建模型
        n_stocks = len(tickers)
        model = StockTransformer(
            n_stocks=n_stocks,
            seq_len=args.seq_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        # 创建掩码，避免关注未来的token
        mask = create_mask(args.seq_len, device) if supports_src_mask else None
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        
        # 训练模型
        logger.info("开始训练模型...")
        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        
        # 计算总参数量
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型总参数数量: {total_params:,}")
        
        # 早停设置
        patience = 10
        no_improve_epochs = 0
        
        for epoch in range(args.epochs):
            # 训练
            train_loss = train_model(model, train_loader, optimizer, criterion, device, mask)
            
            # 验证
            val_loss = evaluate_model(model, val_loader, criterion, device, mask)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f"保存了新的最佳模型，验证损失: {val_loss:.6f}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            
            # 早停
            if epoch > 400 and no_improve_epochs >= patience:
                logger.info(f"连续 {patience} 个epoch没有改善，提前停止训练")
                break
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
        
        # 加载最佳模型
        try:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
            logger.info("成功加载最佳模型")
        except Exception as e:
            logger.warning(f"加载最佳模型时出错: {e}，使用当前模型")
        
        # 使用整个验证集生成最终的关系矩阵
        model.eval()
        relation_matrices = []
        
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch, mask)
                relation_matrices.append(outputs.cpu().numpy())
        
        if relation_matrices:
            # 平均所有批次的关系矩阵
            all_matrices = np.concatenate(relation_matrices, axis=0)
            final_relation_matrix = np.mean(all_matrices, axis=0)
            
            # 保存关系矩阵
            np.save(os.path.join(args.output_dir, 'relation_matrix.npy'), final_relation_matrix)
            
            # 可视化关系矩阵
            plot_relation_matrix(final_relation_matrix, tickers, os.path.join(args.output_dir, 'relation_matrix.png'))
            
            # 输出关系矩阵为CSV
            relation_df = pd.DataFrame(final_relation_matrix, index=tickers, columns=tickers)
            relation_df.to_csv(os.path.join(args.output_dir, 'relation_matrix.csv'))
            
            logger.info("模型训练和评估完成！")
            logger.info(f"结果保存在: {args.output_dir}")
        else:
            logger.error("生成关系矩阵失败，可能是验证集为空")
            
    except Exception as e:
        logger.error(f"运行过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()