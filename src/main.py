import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import os
from model import StockDataset, StockTransformer, train_model
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def get_latest_model_path(checkpoints_dir):
    """Get the path of the latest model from checkpoints directory."""
    model_files = glob.glob(os.path.join(checkpoints_dir, "stock_model_last_update_*.pt"))
    if not model_files:
        return None
    
    # Extract dates from filenames and find the latest
    dates = [datetime.strptime(f.split('_')[-1].split('.')[0], '%Y-%m-%d') for f in model_files]
    latest_idx = np.argmax([d.timestamp() for d in dates])
    return model_files[latest_idx]

def load_and_prepare_data(data_dir, start_date, end_date):
    """Load and prepare stock data."""
    csv_files = glob.glob(f"{data_dir}/*.csv")
    stock_data = {}
    
    for file in csv_files:
        ticker = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Skip empty dataframes
        if df.empty:
            continue
            
        df = df.sort_values('timestamp')
        
        # Forward fill missing data
        df = df.set_index('timestamp').asfreq('D').ffill()
        df = df.reset_index()
        
        stock_data[ticker] = df
    
    return stock_data

def get_next_trading_day(data_dir, current_date, max_look_ahead=5):
    """Find the next day with trading data within a specified range."""
    for days_ahead in range(1, max_look_ahead + 1):
        next_date = current_date + timedelta(days=days_ahead)
        # Try to load data for this date
        data = load_and_prepare_data(data_dir, next_date, next_date)
        
        # If we found data for any stock, consider it a trading day
        if data:
            return next_date
    
    # If no trading day found within range, just return next calendar day
    return current_date + timedelta(days=1)

def train_new_model(data_dir, train_start, train_end, model_params, device):
    """Train a new model using specified date range."""
    # Create datasets
    train_size = int(0.8 * (train_end - train_start).days)
    val_date = train_start + timedelta(days=train_size)
    
    train_dataset = StockDataset(data_dir, train_start, val_date, model_params['seq_len'])
    val_dataset = StockDataset(data_dir, val_date, train_end, model_params['seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_params['batch_size'])
    
    model = StockTransformer(
        seq_len=model_params['seq_len'],
        num_stocks=len(glob.glob(f"{data_dir}/*.csv")),
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    )
    
    train_model(model, train_loader, val_loader, 
                epochs=model_params['epochs'],
                lr=model_params['learning_rate'],
                device=device)
    
    return model

def predict_returns(model, dataset, device):
    """Predict returns for all stocks."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check data availability for the specified date range.")
        
    model.eval()
    with torch.no_grad():
        X, _ = dataset[0]  # Get the latest data point
        X = X.unsqueeze(0).to(device)  # Add batch dimension
        predictions = model(X)
        return predictions.cpu().numpy().squeeze()

def select_stocks(predictions, tickers, softmax_threshold, max_stocks):
    """Select stocks based on predictions and constraints."""
    # Calculate softmax of predictions
    exp_preds = np.exp(predictions)
    softmax_values = exp_preds / exp_preds.sum()
    
    # Sort stocks by predicted returns
    sorted_indices = np.argsort(predictions)[::-1]
    selected_stocks = []
    cumulative_softmax = 0
    
    for idx in sorted_indices:
        if (predictions[idx] <= 0 or 
            len(selected_stocks) >= max_stocks or 
            cumulative_softmax >= softmax_threshold):
            break
            
        selected_stocks.append({
            'ticker': tickers[idx],
            'predicted_return': predictions[idx],
            'weight': softmax_values[idx]
        })
        cumulative_softmax += softmax_values[idx]
    
    return selected_stocks

def plot_portfolio_value(history, initial_capital):
    """Create a line plot of portfolio value over time."""
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    df['value'] = df['value'].astype(float)
    
    # Calculate daily returns and cumulative returns
    df['daily_return'] = df['value'].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod()
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot absolute portfolio value
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x='date', y='value')
    plt.title('Portfolio Value Over Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    
    # Plot cumulative returns
    plt.subplot(2, 1, 2)
    sns.lineplot(data=df, x='date', y='cumulative_return')
    plt.title('Cumulative Returns Over Time')
    plt.ylabel('Cumulative Return (1 = Initial Investment)')
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('portfolio_value.png')
    plt.close()
    
    # Calculate and print performance metrics
    total_return = (df['value'].iloc[-1] - initial_capital) / initial_capital * 100
    annualized_return = ((1 + total_return/100) ** (365/len(df)) - 1) * 100
    sharpe_ratio = np.sqrt(252) * df['daily_return'].mean() / df['daily_return'].std()
    max_drawdown = (df['value'] / df['value'].cummax() - 1).min() * 100
    
    print("\nPerformance Metrics:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Model parameters
    model_params = {
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate
    }
    
    # Initialize portfolio
    portfolio = {
        'cash': args.initial_capital,
        'positions': {},
        'history': []
    }
    
    # Trading simulation
    current_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    last_model_date = None
    
    while current_date <= end_date:
        print(f"\nProcessing date: {current_date.date()}")
        
        # Check if this is a trading day by looking for data
        day_data = load_and_prepare_data(args.data_dir, current_date, current_date)
        
        if not day_data:
            print(f"No trading data available for {current_date.date()}. Skipping to next day.")
            # Record portfolio value (unchanged since no trades)
            if portfolio['history']:
                # Use the last known value
                portfolio['history'].append({
                    'date': current_date,
                    'value': portfolio['history'][-1]['value']
                })
            else:
                # First day, use initial capital
                portfolio['history'].append({
                    'date': current_date,
                    'value': args.initial_capital
                })
                
            # Move to next calendar day
            current_date += timedelta(days=1)
            continue
        
        # Check if we need to train a new model
        latest_model_path = get_latest_model_path(args.checkpoints_dir)
        if latest_model_path is None or (
            last_model_date is not None and 
            (current_date - last_model_date).days >= args.update_interval
        ):
            print("Training new model...")
            train_start = current_date - timedelta(days=args.training_window)
            model = train_new_model(
                args.data_dir, 
                train_start, 
                current_date,
                model_params,
                device
            )
            model_path = os.path.join(
                args.checkpoints_dir,
                f"stock_model_last_update_{current_date.date()}.pt"
            )
            torch.save(model.state_dict(), model_path)
            last_model_date = current_date
        else:
            model = StockTransformer(
                seq_len=model_params['seq_len'],
                num_stocks=len(glob.glob(f"{args.data_dir}/*.csv")),
                d_model=model_params['d_model'],
                num_heads=model_params['num_heads'],
                num_layers=model_params['num_layers'],
                dropout=model_params['dropout']
            )
            model.load_state_dict(torch.load(latest_model_path, weights_only=True))
            model = model.to(device)
            print(f"Loaded existing model from {latest_model_path}")
        
        # Prepare dataset for current date
        dataset_start_date = current_date - timedelta(days=model_params['seq_len'] * 2)
        dataset = StockDataset(
            args.data_dir,
            dataset_start_date,
            current_date,
            model_params['seq_len']
        )
        
        # Get predictions
        predictions = predict_returns(model, dataset, device)
        
        # Select stocks
        tickers = [os.path.basename(f).split('.')[0] for f in glob.glob(f"{args.data_dir}/*.csv")]
        selected_stocks = select_stocks(
            predictions,
            tickers,
            args.softmax_threshold,
            args.max_stocks
        )
        
        # Update portfolio
        # Load current day's stock data - use the data we already loaded
        stock_data = day_data
        
        # First, sell all current positions
        total_value = portfolio['cash']
        for ticker, shares in portfolio['positions'].items():
            if ticker in stock_data and not stock_data[ticker].empty:
                sell_price = stock_data[ticker]['open'].iloc[0]
                total_value += shares * sell_price
                print(f"Sold {shares} shares of {ticker} at ${sell_price:.2f}")
            else:
                print(f"Warning: Cannot sell {ticker} - no data available")
        
        portfolio['cash'] = total_value
        portfolio['positions'] = {}
        
        total_weight = sum(stock['weight'] for stock in selected_stocks)

        # If we have selected stocks with positive predicted returns
        if selected_stocks:
            # Calculate position sizes based on weights
            for stock in selected_stocks:
                ticker = stock['ticker']
                if ticker in stock_data and not stock_data[ticker].empty:
                    weight = stock['weight']
                    stock_price = stock_data[ticker]['open'].iloc[0]
                    
                    # Calculate number of shares to buy
                    position_value = total_value * (weight / total_weight)
                    shares = int(position_value / stock_price)
                    
                    if shares > 0:
                        cost = shares * stock_price
                        if cost <= portfolio['cash']:
                            portfolio['positions'][ticker] = shares
                            portfolio['cash'] -= cost
                            print(f"Bought {shares} shares of {ticker} at ${stock_price:.2f}")
        
        # Calculate total portfolio value for this day
        portfolio_value = portfolio['cash']
        for ticker, shares in portfolio['positions'].items():
            if ticker in stock_data and not stock_data[ticker].empty:
                close_price = stock_data[ticker]['close'].iloc[0]
                portfolio_value += shares * close_price
            else:
                print(f"Warning: Cannot value {ticker} - no data available")
        
        # Record portfolio value
        portfolio['history'].append({
            'date': current_date,
            'value': portfolio_value
        })
        
        # Find and move to next trading day
        next_date = get_next_trading_day(args.data_dir, current_date)
        print(f"Moving to next trading day: {next_date.date()}")
        
        # Fill in portfolio values for skipped non-trading days
        temp_date = current_date + timedelta(days=1)
        while temp_date < next_date:
            portfolio['history'].append({
                'date': temp_date,
                'value': portfolio_value  # Use same value as current trading day
            })
            temp_date += timedelta(days=1)
            
        current_date = next_date
        print(f"Total portfolio value: ${total_value:.2f}")
    
    # Save final results
    results_df = pd.DataFrame(portfolio['history'])
    results_df.to_csv('trading_results.csv', index=False)
    
    # Plot portfolio value and performance metrics
    plot_portfolio_value(portfolio['history'], args.initial_capital)
    
    print("\nTrading simulation completed. Results saved to trading_results.csv")
    print("Portfolio value visualization saved as portfolio_value.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Trading Simulation')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing stock data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--training_window', type=int, default=90, help='Training window in days')
    parser.add_argument('--update_interval', type=int, default=30, help='Model update interval in days')
    parser.add_argument('--softmax_threshold', type=float, default=0.8, help='Cumulative softmax threshold')
    parser.add_argument('--max_stocks', type=int, default=20, help='Maximum number of stocks to hold')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='Initial capital')
    
    # Model parameters
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    main(args)