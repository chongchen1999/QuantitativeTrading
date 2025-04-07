import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Define file paths
base_dir = "/home/tourist/neu/QuantitativeTrading/results/"
files = {
    'Transformer': 'trading_results.csv',
    'Daily Rebalanced': 'daily_rebalanced_results.csv',
    'Momentum': 'momentum_results.csv'
}

# Read data
data = {}
for model, filename in files.items():
    filepath = os.path.join(base_dir, filename)
    df = pd.read_csv(filepath)
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Set date as index
    df.set_index('date', inplace=True)
    data[model] = df

# Create figure
plt.figure(figsize=(14, 8))

# Plot each model's performance
for model, df in data.items():
    plt.plot(df.index, df['value'], label=model, linewidth=2)

# Calculate and display returns
for model, df in data.items():
    initial_value = df['value'].iloc[0]
    final_value = df['value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    print(f"{model} total return: {total_return:.2f}%")

# Add title and labels
plt.title('Comparison of Trading Models Performance (2020-2023)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Portfolio Value', fontsize=14)
plt.grid(True, alpha=0.3)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Add legend
plt.legend(fontsize=12)

# Adjust layout
plt.tight_layout()

# Save and show plot
plt.savefig('trading_models_comparison.png', dpi=300)
plt.show()