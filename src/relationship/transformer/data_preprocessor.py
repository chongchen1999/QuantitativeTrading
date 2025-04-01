import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="股票数据预处理工具")
    parser.add_argument("--data_dir", type=str, required=True, help="股票CSV文件所在目录")
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="处理后数据输出目录")
    parser.add_argument("--visualize", action="store_true", help="是否可视化数据")
    return parser.parse_args()

def load_stock_data(data_dir):
    """加载目录中的所有股票CSV文件"""
    print(f"正在从 {data_dir} 加载股票数据...")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"错误: 在 {data_dir} 中没有找到CSV文件")
        return {}
    
    # 存储股票数据的字典
    stock_data = {}
    
    for file_path in csv_files:
        ticker = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 确保必要的列存在
            required_columns = ['timestamp', 'close']
            if not all(col in df.columns for col in required_columns):
                # 检查是否有可能的替代列
                column_mapping = {
                    'timestamp': ['date', 'time', 'datetime', 'Date', 'Time', 'DateTime'],
                    'close': ['Close', 'closing_price', 'closing', 'close_price']
                }
                
                # 尝试重命名列
                for req_col, alt_cols in column_mapping.items():
                    if req_col not in df.columns:
                        for alt_col in alt_cols:
                            if alt_col in df.columns:
                                df = df.rename(columns={alt_col: req_col})
                                break
            
            # 再次检查必要的列是否存在
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                print(f"警告: {ticker} 数据缺少必要的列: {missing_cols}，跳过此文件")
                continue
            
            # 转换timestamp为日期格式
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                print(f"警告: {ticker} 数据中的timestamp列无法转换为日期格式: {e}，尝试其他格式")
                # 尝试不同的日期格式
                date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y']
                for date_format in date_formats:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format=date_format)
                        break
                    except:
                        continue
                
                # 如果仍然无法转换
                if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    print(f"错误: {ticker} 数据中的timestamp列无法转换为日期格式，跳过此文件")
                    continue
            
            # 确保close列是数值类型
            try:
                df['close'] = pd.to_numeric(df['close'])
            except Exception as e:
                print(f"警告: {ticker} 数据中的close列无法转换为数值类型: {e}，尝试清理")
                # 尝试清理数据
                df['close'] = df['close'].replace('[\$,]', '', regex=True)
                try:
                    df['close'] = pd.to_numeric(df['close'])
                except:
                    print(f"错误: {ticker} 数据中的close列无法转换为数值类型，跳过此文件")
                    continue
            
            # 排序数据
            df = df.sort_values('timestamp')
            
            # 保存处理后的数据
            stock_data[ticker] = df
            print(f"成功加载 {ticker} 数据，共 {len(df)} 行")
            
        except Exception as e:
            print(f"错误: 处理 {ticker} 数据时出错: {e}")
    
    print(f"共加载了 {len(stock_data)} 支股票的数据")
    return stock_data

def calculate_log_returns(stock_data):
    """计算每支股票的对数收益率"""
    print("计算每支股票的对数收益率...")
    
    log_returns = {}
    
    for ticker, df in stock_data.items():
        # 创建一个副本
        lr_df = df.copy()
        
        # 计算对数收益率
        lr_df['log_return'] = np.log(lr_df['close'] / lr_df['close'].shift(1))
        
        # 删除第一行（无法计算收益率）
        lr_df = lr_df.dropna(subset=['log_return'])
        
        log_returns[ticker] = lr_df
        
        print(f"{ticker} 对数收益率计算完成，共 {len(lr_df)} 行")
    
    return log_returns

def align_timestamps(log_returns):
    """确保所有股票数据使用相同的时间戳"""
    print("对齐所有股票的时间戳...")
    
    # 获取所有时间戳
    all_timestamps = set()
    for df in log_returns.values():
        all_timestamps.update(df['timestamp'].tolist())
    
    # 按时间排序
    all_timestamps = sorted(list(all_timestamps))
    
    print(f"总共有 {len(all_timestamps)} 个不同的交易日")
    
    # 创建一个包含所有时间戳的DataFrame
    aligned_data = pd.DataFrame({'timestamp': all_timestamps})
    
    # 将每支股票的log_return添加到DataFrame中
    for ticker, df in log_returns.items():
        # 创建一个以timestamp为索引的Series
        ticker_series = pd.Series(df['log_return'].values, index=df['timestamp'])
        # 重新索引以包含所有时间戳，NaN表示缺失值
        aligned_data[ticker] = aligned_data['timestamp'].map(lambda x: ticker_series.get(x, np.nan))
    
    # 计算每个时间戳的非NaN值数量
    aligned_data['valid_count'] = aligned_data.drop('timestamp', axis=1).notna().sum(axis=1)
    
    # 计算每支股票的非NaN值数量
    valid_counts = aligned_data.drop(['timestamp', 'valid_count'], axis=1).notna().sum()
    valid_counts = valid_counts.sort_values(ascending=False)
    
    print("每支股票的有效数据天数:")
    for ticker, count in valid_counts.items():
        print(f"{ticker}: {count} 天 ({count/len(all_timestamps):.2%})")
    
    return aligned_data

def filter_aligned_data(aligned_data, min_valid_ratio=0.9):
    """过滤对齐后的数据，去除缺失值过多的行和列"""
    print(f"过滤对齐后的数据，要求至少 {min_valid_ratio:.0%} 的数据有效...")
    
    # 总列数（不包括timestamp和valid_count）
    n_stocks = aligned_data.shape[1] - 2
    
    # 根据valid_count过滤行
    min_valid_count = int(n_stocks * min_valid_ratio)
    filtered_data = aligned_data[aligned_data['valid_count'] >= min_valid_count].copy()
    
    print(f"过滤前: {len(aligned_data)} 行，过滤后: {len(filtered_data)} 行")
    
    # 计算每列的有效值比例
    valid_ratios = filtered_data.drop(['timestamp', 'valid_count'], axis=1).notna().mean()
    
    # 根据有效值比例过滤列
    keep_columns = ['timestamp'] + list(valid_ratios[valid_ratios >= min_valid_ratio].index)
    filtered_data = filtered_data[keep_columns]
    
    print(f"过滤前: {n_stocks} 支股票，过滤后: {len(keep_columns)-1} 支股票")
    
    # 删除valid_count列
    if 'valid_count' in filtered_data.columns:
        filtered_data = filtered_data.drop('valid_count', axis=1)
    
    return filtered_data

def fill_missing_values(filtered_data):
    """填充缺失值"""
    print("填充缺失值...")
    
    # 使用前向填充
    filled_data = filtered_data.copy()
    filled_data = filled_data.set_index('timestamp')
    filled_data = filled_data.fillna(method='ffill')
    
    # 使用后向填充处理仍然存在的缺失值
    filled_data = filled_data.fillna(method='bfill')
    
    # 重置索引
    filled_data = filled_data.reset_index()
    
    # 检查是否还有缺失值
    missing_count = filled_data.isna().sum().sum()
    if missing_count > 0:
        print(f"警告: 填充后仍然有 {missing_count} 个缺失值")
    else:
        print("所有缺失值已填充")
    
    return filled_data

def visualize_data(stock_data, log_returns, aligned_data, processed_data, output_dir):
    """可视化数据"""
    print("生成数据可视化...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 可视化每支股票的收盘价
    for ticker, df in stock_data.items():
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['close'])
        plt.title(f"{ticker} 收盘价")
        plt.xlabel("日期")
        plt.ylabel("价格")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{ticker}_close_price.png"))
        plt.close()
    
    # 2. 可视化每支股票的对数收益率
    for ticker, df in log_returns.items():
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['log_return'])
        plt.title(f"{ticker} 对数收益率")
        plt.xlabel("日期")
        plt.ylabel("对数收益率")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{ticker}_log_return.png"))
        plt.close()
    
    # 3. 可视化有效数据计数
    if 'valid_count' in aligned_data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(aligned_data['timestamp'], aligned_data['valid_count'])
        plt.title("每个交易日的有效数据计数")
        plt.xlabel("日期")
        plt.ylabel("有效数据计数")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "valid_data_count.png"))
        plt.close()
    
    # 4. 可视化相关性热图
    if len(processed_data.columns) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = processed_data.drop('timestamp', axis=1).corr()
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title("股票对数收益率相关性热图")
        
        # 添加刻度标签
        tickers = corr_matrix.columns
        plt.xticks(np.arange(len(tickers)), tickers, rotation=90)
        plt.yticks(np.arange(len(tickers)), tickers)
        
        # 添加文本标注
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', 
                         color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
    
    print(f"数据可视化完成，结果保存在 {output_dir}")

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载股票数据
    stock_data = load_stock_data(args.data_dir)
    if not stock_data:
        print("没有加载到有效的股票数据，退出程序")
        return
    
    # 计算对数收益率
    log_returns = calculate_log_returns(stock_data)
    
    # 对齐时间戳
    aligned_data = align_timestamps(log_returns)
    
    # 过滤数据
    filtered_data = filter_aligned_data(aligned_data)
    
    # 填充缺失值
    processed_data = fill_missing_values(filtered_data)
    
    # 输出处理后的数据
    processed_data.to_csv(os.path.join(args.output_dir, "processed_data.csv"), index=False)
    print(f"处理后的数据已保存到 {os.path.join(args.output_dir, 'processed_data.csv')}")
    
    # 可视化数据
    if args.visualize:
        visualize_data(stock_data, log_returns, aligned_data, processed_data, os.path.join(args.output_dir, "visualizations"))
    
    print("数据预处理完成！")

if __name__ == "__main__":
    main()