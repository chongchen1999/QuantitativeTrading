# Transformer-Based Deep Learning for Quantitative Trading

This repository contains a complete implementation of a transformer-based deep learning model for quantitative trading. The approach leverages recent advances in attention mechanisms and sequential modeling to predict stock returns and make trading decisions.

## Project Overview

This project implements a complete quantitative trading system using transformer neural networks to predict stock price movements. The system includes:

1. A custom-designed transformer architecture for time series financial data
2. Comprehensive data preprocessing and feature engineering
3. An end-to-end trading simulation framework
4. Benchmark strategies for performance comparison

## Model Architecture

The core of the system is a transformer-based neural network designed specifically for financial time series:

- Multi-head self-attention layers to capture complex temporal patterns
- Positional encoding to preserve sequential information
- Custom feature embeddings for financial data

The model captures both within-stock temporal patterns and cross-sectional relationships between different stocks.

## Key Components

### Data Processing

- `StockDataset` class for loading and preprocessing financial time series
- Normalization and feature engineering of raw stock data
- Time-aligned data preparation across multiple stocks

### Transformer Model 

- `StockTransformer` implements the core prediction model
- Custom positional encoding for financial time series
- Multi-head attention mechanisms to capture complex temporal dependencies

### Trading Simulation

- A comprehensive backtesting framework that simulates real trading conditions
- Position sizing based on model predictions
- Portfolio rebalancing with customizable frequency
- Detailed performance tracking and visualization

### Benchmark Strategies

Two benchmark strategies are included for performance comparison:

1. `benchmark.py`: Equal-weight portfolio with daily rebalancing
2. `benchmark_momentum.py`: Momentum-based portfolio strategy with configurable lookback periods

## Data Selection

The repository includes a data preprocessing script (`data_preprocessing.py`) which:

- Selects a representative sample of stocks based on return distribution
- Ensures data quality and completeness
- Creates a balanced dataset across different return profiles

## Usage

### Prerequisites

- PyTorch
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Training a Model

```bash
python main.py --data_dir path/to/stock/data --start_date 2021-01-01 --end_date 2022-01-01 --checkpoints_dir ./checkpoints --training_window 90 --update_interval 30 --seq_len 20
```

### Running Benchmarks

```bash
# Equal-weight benchmark
python benchmark.py --data_dir path/to/stock/data --start_date 2021-01-01 --end_date 2022-01-01 --initial_capital 1000000

# Momentum benchmark
python benchmark_momentum.py --data_dir path/to/stock/data --start_date 2021-01-01 --end_date 2022-01-01 --initial_capital 1000000 --rebalance_days 14
```

### Data Preprocessing

```bash
python data_preprocessing.py
```

## Command Line Arguments

### Main Model (main.py)

| Argument | Description |
|----------|-------------|
| `--data_dir` | Directory containing stock data (CSV files) |
| `--checkpoints_dir` | Directory for model checkpoints |
| `--start_date` | Simulation start date (YYYY-MM-DD) |
| `--end_date` | Simulation end date (YYYY-MM-DD) |
| `--training_window` | Training window in days |
| `--update_interval` | Model update interval in days |
| `--softmax_threshold` | Cumulative softmax threshold for stock selection |
| `--max_stocks` | Maximum number of stocks to hold |
| `--initial_capital` | Initial capital for simulation |
| `--seq_len` | Sequence length for time series |
| `--batch_size` | Training batch size |
| `--d_model` | Model dimension |
| `--num_heads` | Number of attention heads |
| `--num_layers` | Number of transformer layers |
| `--dropout` | Dropout rate |
| `--epochs` | Number of training epochs |
| `--learning_rate` | Learning rate |

### Benchmark Strategies

Similar command-line arguments for specifying data location, simulation period, and trading parameters.

## Performance Metrics

The system tracks and visualizes multiple performance metrics:

- Total return
- Annualized return
- Sharpe ratio
- Maximum drawdown
- Daily performance
- Cumulative performance

## Output

The model and benchmark strategies generate:

- Detailed performance logs
- Portfolio value CSV files
- Performance visualization graphs
- Trading activity summaries

## Model Comparison

To compare the transformer model against benchmarks:

1. Run all three strategies with the same parameters:
   - Start/end dates
   - Initial capital
   - Stock universe

2. Compare the generated performance metrics and visualizations
