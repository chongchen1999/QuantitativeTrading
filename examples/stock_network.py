import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso

# 假设我们有 10 只股票的历史收益率数据
np.random.seed(42)
returns = np.random.randn(100, 10)  # 100 天，10 只股票

# 计算普通协方差矩阵
cov_matrix = np.cov(returns.T)
print("普通协方差矩阵:")
print(np.round(cov_matrix, 4))  # 四舍五入到4位小数

# 计算相关系数矩阵
corr_matrix = np.corrcoef(returns.T)
print("\n相关系数矩阵:")
print(np.round(corr_matrix, 4))

# 计算 Graphical Lasso 估计的精度矩阵
model = GraphicalLasso(alpha=0.1)
model.fit(returns)
precision_matrix = model.precision_
print("\nGraphical Lasso 精度矩阵 (协方差矩阵的逆):")
print(np.round(precision_matrix, 4))

# 可视化普通协方差矩阵
plt.figure(figsize=(12, 4))
plt.subplot(131)
sns.heatmap(cov_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("协方差矩阵")

# 可视化相关系数矩阵
plt.subplot(132)
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("相关系数矩阵")

# 可视化精度矩阵
plt.subplot(133)
sns.heatmap(precision_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.title("精度矩阵")

plt.tight_layout()
plt.show()

# 输出股票对之间的关系强度
print("\n股票对之间的关系强度 (基于协方差):")
stock_names = [f"股票{i+1}" for i in range(10)]
for i in range(10):
    for j in range(i+1, 10):
        print(f"{stock_names[i]} 和 {stock_names[j]}: {cov_matrix[i, j]:.4f}")