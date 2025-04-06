import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
import argparse
import logging
import sys

# Set up logging
def setup_logging(log_file='benchmark_simulation.log'):
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_and_prepare_data(data_dir, start_date, end_date):
    """Load and prepare stock data."""
    logger.info(f"Loading stock data from {data_dir} for period {start_date.date()} to {end_date.date()}")
    
    csv_files = glob.glob(f"{data_dir}/*.csv")
    logger.info(f"Found {len(csv_files)} stock data files")
    
    stock_data = {}
    
    for file in csv_files:
        ticker = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Skip empty dataframes
        if df.empty:
            logger.warning(f"No data available for {ticker} in the specified date range")
            continue
            
        df = df.sort_values('timestamp')
        
        # Forward fill missing data
        df = df.set_index('timestamp').asfreq('D').ffill()
        df = df.reset_index()
        
        stock_data[ticker] = df
        logger.debug(f"Loaded {len(df)} days of data for {ticker}")
    
    logger.info(f"Successfully loaded data for {len(stock_data)} stocks")
    return stock_data

def plot_portfolio_value(history, initial_capital):
    """Create a line plot of portfolio value over time."""
    logger.info("Generating portfolio value plots and calculating performance metrics")
    
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
    plt.savefig('benchmark_portfolio_value.png')
    plt.close()
    
    logger.info("Portfolio visualization saved as benchmark_portfolio_value.png")
    
    # Calculate and print performance metrics
    total_return = (df['value'].iloc[-1] - initial_capital) / initial_capital * 100
    annualized_return = ((1 + total_return/100) ** (365/len(df)) - 1) * 100
    sharpe_ratio = np.sqrt(252) * df['daily_return'].mean() / df['daily_return'].std()
    max_drawdown = (df['value'] / df['value'].cummax() - 1).min() * 100
    
    logger.info("\nDaily Rebalanced Strategy Performance Metrics:")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Annualized Return: {annualized_return:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")

def main(args):
    global logger
    logger = setup_logging(os.path.join(os.path.dirname(args.data_dir), 'benchmark_simulation.log'))
    
    logger.info("Starting Daily Rebalanced Equal-Weight Portfolio Simulation")
    logger.info(f"Arguments: {args}")
    
    # Process dates
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    logger.info(f"Running daily rebalanced portfolio from {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial capital: ${args.initial_capital:.2f}")
    
    # Load all stock data
    stock_data = load_and_prepare_data(args.data_dir, start_date, end_date)
    
    if not stock_data:
        logger.error("No stock data available for the specified date range.")
        return
    
    # Get unique trading days across all stocks
    all_dates = set()
    for ticker, df in stock_data.items():
        all_dates.update(df['timestamp'].dt.date)
    
    all_dates = sorted(all_dates)
    logger.info(f"Found {len(all_dates)} unique trading days in the data")
    
    # Initialize portfolio
    initial_capital = args.initial_capital
    
    # Portfolio tracking
    portfolio = {
        'cash': initial_capital,
        'positions': {},
        'history': []
    }
    
    logger.info("Starting portfolio simulation...")
    
    # Simulate trading for each day
    for current_date in all_dates:
        current_datetime = pd.Timestamp(current_date)
        logger.info(f"Processing date: {current_date}")
        
        # Get data for current trading day
        day_data = {}
        for ticker, df in stock_data.items():
            df_day = df[df['timestamp'].dt.date == current_date]
            if not df_day.empty:
                day_data[ticker] = df_day.iloc[0]
        
        # Skip if no data for this day
        if not day_data:
            logger.warning(f"No data available for {current_date}. Skipping.")
            continue
            
        # Calculate current portfolio value
        portfolio_value = portfolio['cash']
        for ticker, shares in portfolio['positions'].items():
            if ticker in day_data:
                open_price = day_data[ticker]['open']
                portfolio_value += shares * open_price
        
        logger.info(f"Date: {current_date}, Portfolio Value: ${portfolio_value:.2f}")
        
        # Sell all current positions to reset
        for ticker, shares in list(portfolio['positions'].items()):
            if ticker in day_data and shares > 0:
                open_price = day_data[ticker]['open']
                sell_value = shares * open_price
                portfolio['cash'] += sell_value
                portfolio['positions'][ticker] = 0
                logger.debug(f"Sold {shares} shares of {ticker} at ${open_price:.2f}, value: ${sell_value:.2f}")
        
        # Distribute money equally across available stocks
        available_stocks = list(day_data.keys())
        num_stocks = len(available_stocks)
        
        if num_stocks == 0:
            logger.warning(f"No stocks available for trading on {current_date}. Skipping.")
            continue
            
        # Allocate cash equally
        per_stock_capital = portfolio['cash'] / num_stocks
        logger.debug(f"Allocating ${per_stock_capital:.2f} to each of {num_stocks} stocks")
        
        # Buy new positions
        total_invested = 0
        for ticker in available_stocks:
            price = day_data[ticker]['open']
            if price > 0:
                shares = int(per_stock_capital / price)
                cost = shares * price
                portfolio['positions'][ticker] = shares
                total_invested += cost
                logger.debug(f"Bought {shares} shares of {ticker} at ${price:.2f}, cost: ${cost:.2f}")
        
        # Update remaining cash
        portfolio['cash'] = portfolio_value - total_invested
        logger.debug(f"Remaining cash: ${portfolio['cash']:.2f}")
        
        # Calculate end-of-day portfolio value
        end_day_value = portfolio['cash']
        for ticker, shares in portfolio['positions'].items():
            if ticker in day_data:
                close_price = day_data[ticker]['close']
                end_day_value += shares * close_price
        
        logger.debug(f"End-of-day portfolio value: ${end_day_value:.2f}")
        
        # Record portfolio value
        portfolio['history'].append({
            'date': current_datetime,
            'value': end_day_value
        })
    
    logger.info("Trading simulation completed. Processing results...")
    
    # Fill in non-trading days for continuous visualization
    all_days = []
    current_date = start_date
    while current_date <= end_date:
        all_days.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Filling in values for non-trading days ({len(all_days)} total days)")
    
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
    logger.info("Results saved to daily_rebalanced_results.csv")
    
    # Plot portfolio value and performance metrics
    plot_portfolio_value(continuous_history, initial_capital)
    
    logger.info("\nDaily rebalanced portfolio simulation completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Daily Rebalanced Equal-Weight Portfolio Simulation')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing stock data')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='Initial capital')
    
    args = parser.parse_args()
    main(args)