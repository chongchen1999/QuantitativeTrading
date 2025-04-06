import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 读取数据文件夹中的所有CSV文件
data_path = '/home/tourist/neu/QuantitativeTrading/data/dataset'
all_files = os.listdir(data_path)
csv_files = [f for f in all_files if f.endswith('.csv')]

# 初始设置
initial_total = 1000000  # 初始总资金
per_stock_investment = 5000  # 每支股票投资金额
start_date = '2020-09-01'
end_date = '2023-09-01'

# 存储每支股票的每日价值
all_daily_values = []

# 处理每个文件
for csv_file in csv_files:
    try:
        # 读取CSV文件
        df = pd.read_csv(os.path.join(data_path, csv_file))
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 过滤时间范围
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        df = df[mask]
        
        if len(df) == 0:
            continue
            
        # 计算初始股数（用收盘价）
        initial_price = df.iloc[0]['close']
        shares = per_stock_investment / initial_price
        
        # 计算每日价值
        df['daily_value'] = shares * df['close']
        
        # 只保留需要的列
        value_series = df[['timestamp', 'daily_value']].copy()
        all_daily_values.append(value_series)
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

# 合并所有股票的每日价值
if all_daily_values:
    # 合并所有数据
    combined_values = pd.concat(all_daily_values)
    
    # 按日期分组并求和
    portfolio_value = combined_values.groupby('timestamp')['daily_value'].sum().reset_index()
    portfolio_value = portfolio_value.sort_values('timestamp')
    
    # 计算一些基本统计信息
    start_value = portfolio_value['daily_value'].iloc[0]
    end_value = portfolio_value['daily_value'].iloc[-1]
    total_return = ((end_value / start_value) - 1) * 100
    
    print("\nPortfolio Statistics:")
    print(f"Start Value: ${start_value:,.2f}")
    print(f"End Value: ${end_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # 创建收益曲线图
    plt.figure(figsize=(15, 8))
    plt.plot(portfolio_value['timestamp'], portfolio_value['daily_value'], 
             linewidth=2, color='blue')
    
    # 设置图表标题和标签
    plt.title('Portfolio Value Over Time (2020-2023)', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    
    # 格式化y轴为货币格式
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 旋转x轴日期标签以防重叠
    plt.xticks(rotation=45)
    
    # 调整布局以确保所有元素可见
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('portfolio_value.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
else:
    print("No data was processed successfully")