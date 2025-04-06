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
def setup_logging(log_file='momentum_benchmark_simulation.log'):
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

def calculate_momentum_scores(stock_data, current_date, lookback_periods):
    """
    Calculate momentum scores for each stock based on their performance over multiple periods.
    
    Parameters:
    - stock_data: Dictionary of dataframes with stock price data
    - current_date: Current simulation date
    - lookback_periods: Dictionary with period names as keys and (days, weight) tuples as values
    
    Returns:
    - Dictionary of momentum scores for each stock
    """
    momentum_scores = {}
    
    for ticker, df in stock_data.items():
        # Filter data up to the current date
        df_hist = df[df['timestamp'].dt.date <= current_date]
        
        if len(df_hist) < max([days for days, _ in lookback_periods.values()]):
            # Not enough historical data
            continue
            
        # Calculate returns for different periods
        weighted_returns = 0
        total_weight = 0
        
        for period_name, (days, weight) in lookback_periods.items():
            if len(df_hist) >= days:
                # Calculate return for this period
                current_price = df_hist.iloc[-1]['close']
                past_price = df_hist.iloc[-days]['close'] if days < len(df_hist) else df_hist.iloc[0]['close']
                period_return = (current_price / past_price) - 1
                
                # Add weighted return
                weighted_returns += period_return * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            momentum_scores[ticker] = weighted_returns / total_weight
    
    return momentum_scores

def allocate_capital_by_momentum(momentum_scores, available_capital, min_stocks=5, max_allocation=0.2):
    """
    Allocate capital based on momentum scores.
    
    Parameters:
    - momentum_scores: Dictionary of momentum scores by ticker
    - available_capital: Total capital to allocate
    - min_stocks: Minimum number of stocks to include
    - max_allocation: Maximum allocation to a single stock (as percentage)
    
    Returns:
    - Dictionary with ticker as key and allocation amount as value
    """
    if not momentum_scores:
        return {}
        
    # Sort stocks by momentum score
    sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Determine how many stocks to include (at least min_stocks, or all if fewer available)
    num_stocks = max(min_stocks, min(len(sorted_stocks), int(len(sorted_stocks) * 0.5)))
    top_stocks = sorted_stocks[:num_stocks]
    
    # Filter out negative momentum stocks unless we don't have enough positive ones
    positive_momentum = [s for s in top_stocks if s[1] > 0]
    if len(positive_momentum) >= min_stocks:
        top_stocks = positive_momentum
    
    # Ensure we have at least some stocks
    if not top_stocks:
        top_stocks = sorted_stocks[:min_stocks]
    
    # Calculate weights based on relative momentum
    total_score = sum(max(0.01, score) for _, score in top_stocks)  # Ensure non-zero total
    weights = {ticker: max(0.01, score) / total_score for ticker, score in top_stocks}
    
    # Apply maximum allocation constraint
    for ticker in weights:
        weights[ticker] = min(weights[ticker], max_allocation)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:  # Avoid division by zero
        weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
    
    # Convert weights to capital allocations
    allocations = {ticker: weight * available_capital for ticker, weight in weights.items()}
    
    return allocations

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
    plt.title('Momentum Strategy Portfolio Value Over Time')
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
    plt.savefig('momentum_portfolio_value.png')
    plt.close()
    
    logger.info("Portfolio visualization saved as momentum_portfolio_value.png")
    
    # Calculate and print performance metrics
    total_return = (df['value'].iloc[-1] - initial_capital) / initial_capital * 100
    annualized_return = ((1 + total_return/100) ** (365/len(df)) - 1) * 100
    sharpe_ratio = np.sqrt(252) * df['daily_return'].mean() / df['daily_return'].std()
    max_drawdown = (df['value'] / df['value'].cummax() - 1).min() * 100
    
    logger.info("\nMomentum Strategy Performance Metrics:")
    logger.info(f"Total Return: {total_return:.2f}%")
    logger.info(f"Annualized Return: {annualized_return:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Save performance metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Sharpe Ratio', 'Maximum Drawdown (%)'],
        'Value': [total_return, annualized_return, sharpe_ratio, max_drawdown]
    })
    metrics_df.to_csv('momentum_performance_metrics.csv', index=False)
    logger.info("Performance metrics saved to momentum_performance_metrics.csv")

def main(args):
    global logger
    logger = setup_logging(os.path.join(os.path.dirname(args.data_dir), 'momentum_benchmark_simulation.log'))
    
    logger.info("Starting Momentum-Based Portfolio Simulation")
    logger.info(f"Arguments: {args}")
    
    # Process dates
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    
    # Define momentum lookback periods and their weights
    lookback_periods = {
        'short_term': (7, 0.25),     # 1 week, 25% weight
        'medium_term': (30, 0.5),    # 1 month, 50% weight
        'long_term': (90, 0.25)      # 3 months, 25% weight
    }
    
    # Get rebalancing frequency
    rebalance_days = args.rebalance_days
    
    logger.info(f"Running momentum portfolio from {start_date.date()} to {end_date.date()}")
    logger.info(f"Initial capital: ${args.initial_capital:.2f}")
    logger.info(f"Rebalancing frequency: Every {rebalance_days} days")
    logger.info(f"Momentum periods: {lookback_periods}")
    
    # Load all stock data
    stock_data = load_and_prepare_data(args.data_dir, start_date - timedelta(days=max([days for days, _ in lookback_periods.values()])), end_date)
    
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
        'history': [],
        'last_rebalance_date': None
    }
    
    logger.info("Starting portfolio simulation...")
    
    # Simulate trading for each day
    for current_date in all_dates:
        current_datetime = pd.Timestamp(current_date)
        logger.debug(f"Processing date: {current_date}")
        
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
        
        # Determine if rebalancing is needed
        days_since_last_rebalance = float('inf')
        if portfolio['last_rebalance_date'] is not None:
            days_since_last_rebalance = (current_datetime.date() - portfolio['last_rebalance_date']).days
        
        should_rebalance = (portfolio['last_rebalance_date'] is None or 
                           days_since_last_rebalance >= rebalance_days)
        
        if should_rebalance:
            logger.info(f"Date: {current_date}, Portfolio Value: ${portfolio_value:.2f} - REBALANCING")
            
            # Calculate momentum scores
            momentum_scores = calculate_momentum_scores(stock_data, current_date, lookback_periods)
            
            if momentum_scores:
                # Allocate capital based on momentum
                allocations = allocate_capital_by_momentum(
                    momentum_scores, 
                    portfolio_value,
                    min_stocks=args.min_stocks,
                    max_allocation=args.max_allocation
                )
                
                # Sell all current positions
                for ticker, shares in list(portfolio['positions'].items()):
                    if ticker in day_data and shares > 0:
                        open_price = day_data[ticker]['open']
                        sell_value = shares * open_price
                        portfolio['cash'] += sell_value
                        portfolio['positions'][ticker] = 0
                        logger.debug(f"Sold {shares} shares of {ticker} at ${open_price:.2f}, value: ${sell_value:.2f}")
                
                # Buy new positions based on allocations
                for ticker, allocation in allocations.items():
                    if ticker in day_data:
                        price = day_data[ticker]['open']
                        if price > 0:
                            shares = int(allocation / price)
                            cost = shares * price
                            portfolio['positions'][ticker] = shares
                            portfolio['cash'] -= cost
                            logger.debug(f"Bought {shares} shares of {ticker} at ${price:.2f}, cost: ${cost:.2f}")
                
                # Update last rebalance date
                portfolio['last_rebalance_date'] = current_datetime.date()
            else:
                logger.warning("Unable to calculate momentum scores. No rebalancing performed.")
        else:
            logger.debug(f"Date: {current_date}, Portfolio Value: ${portfolio_value:.2f}")
        
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
    results_df.to_csv('momentum_results.csv', index=False)
    logger.info("Results saved to momentum_results.csv")
    
    # Plot portfolio value and performance metrics
    plot_portfolio_value(continuous_history, initial_capital)
    
    logger.info("\nMomentum portfolio simulation completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Momentum-Based Portfolio Simulation')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing stock data')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='Initial capital')
    parser.add_argument('--rebalance_days', type=int, default=14, help='Rebalancing frequency in days')
    parser.add_argument('--min_stocks', type=int, default=5, help='Minimum number of stocks in portfolio')
    parser.add_argument('--max_allocation', type=float, default=0.2, help='Maximum allocation to a single stock (0-1)')
    
    args = parser.parse_args()
    main(args)