import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
import argparse

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
    plt.title('Daily Rebalanced Equal-Weight Portfolio Value Over Time')
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
    plt.savefig('daily_rebalanced_portfolio_value.png')
    plt.close()
    
    # Calculate and print performance metrics
    total_return = (df['value'].iloc[-1] - initial_capital) / initial_capital * 100
    annualized_return = ((1 + total_return/100) ** (365/len(df)) - 1) * 100
    sharpe_ratio = np.sqrt(252) * df['daily_return'].mean() / df['daily_return'].std()
    max_drawdown = (df['value'] / df['value'].cummax() - 1).min() * 100
    
    print("\nDaily Rebalanced Strategy Performance Metrics:")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

def main(args):
    # Process dates
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    print(f"Running daily rebalanced portfolio from {start_date.date()} to {end_date.date()}")
    
    # Load all stock data
    stock_data = load_and_prepare_data(args.data_dir, start_date, end_date)
    
    if not stock_data:
        print("No stock data available for the specified date range.")
        return
    
    # Get unique trading days across all stocks
    all_dates = set()
    for ticker, df in stock_data.items():
        all_dates.update(df['timestamp'].dt.date)
    
    all_dates = sorted(all_dates)
    
    # Initialize portfolio
    initial_capital = args.initial_capital
    
    # Portfolio tracking
    portfolio = {
        'cash': initial_capital,
        'positions': {},
        'history': []
    }
    
    # Simulate trading for each day
    for current_date in all_dates:
        current_datetime = pd.Timestamp(current_date)
        
        # Get data for current trading day
        day_data = {}
        for ticker, df in stock_data.items():
            df_day = df[df['timestamp'].dt.date == current_date]
            if not df_day.empty:
                day_data[ticker] = df_day.iloc[0]
        
        # Skip if no data for this day
        if not day_data:
            continue
            
        # Calculate current portfolio value
        portfolio_value = portfolio['cash']
        for ticker, shares in portfolio['positions'].items():
            if ticker in day_data:
                open_price = day_data[ticker]['open']
                portfolio_value += shares * open_price
        
        # Output how much money we have today
        print(f"Date: {current_date}, Portfolio Value: ${portfolio_value:.2f}")
        
        # Sell all current positions to reset
        for ticker, shares in list(portfolio['positions'].items()):
            if ticker in day_data:
                open_price = day_data[ticker]['open']
                portfolio['cash'] += shares * open_price
                portfolio['positions'][ticker] = 0
        
        # Distribute money equally across available stocks
        available_stocks = list(day_data.keys())
        num_stocks = len(available_stocks)
        
        if num_stocks == 0:
            continue
            
        # Allocate cash equally
        per_stock_capital = portfolio['cash'] / num_stocks
        
        # Buy new positions
        total_invested = 0
        for ticker in available_stocks:
            price = day_data[ticker]['open']
            if price > 0:
                shares = int(per_stock_capital / price)
                cost = shares * price
                portfolio['positions'][ticker] = shares
                total_invested += cost
        
        # Update remaining cash
        portfolio['cash'] = portfolio_value - total_invested
        
        # Calculate end-of-day portfolio value
        end_day_value = portfolio['cash']
        for ticker, shares in portfolio['positions'].items():
            if ticker in day_data:
                close_price = day_data[ticker]['close']
                end_day_value += shares * close_price
        
        # Record portfolio value
        portfolio['history'].append({
            'date': current_datetime,
            'value': end_day_value
        })
    
    # Fill in non-trading days for continuous visualization
    all_days = []
    current_date = start_date
    while current_date <= end_date:
        all_days.append(current_date)
        current_date += timedelta(days=1)
    
    # Create a continuous timeline including non-trading days
    continuous_history = []
    trading_day_values = {h['date'].date(): h['value'] for h in portfolio['history']}
    
    last_value = initial_capital
    for day in all_days:
        if day.date() in trading_day_values:
            last_value = trading_day_values[day.date()]
        
        continuous_history.append({
            'date': day,
            'value': last_value
        })
    
    # Save results
    results_df = pd.DataFrame(continuous_history)
    results_df.to_csv('daily_rebalanced_results.csv', index=False)
    
    # Plot portfolio value and performance metrics
    plot_portfolio_value(continuous_history, initial_capital)
    
    print("\nDaily rebalanced portfolio simulation completed. Results saved to daily_rebalanced_results.csv")
    print("Portfolio value visualization saved as daily_rebalanced_portfolio_value.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Daily Rebalanced Equal-Weight Portfolio Simulation')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing stock data')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='Initial capital')
    
    args = parser.parse_args()
    main(args)