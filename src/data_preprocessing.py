import pandas as pd
import os
from datetime import datetime
import shutil
import numpy as np
from scipy import stats

def load_and_validate_stock(file_path, start_date, end_date):
    """
    Load stock data and validate if it meets the date range criteria.
    Returns DataFrame if valid, None otherwise.
    """
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Check if data covers the required date range
        data_start = df['timestamp'].min()
        data_end = df['timestamp'].max()
        
        if data_start <= start_date and data_end >= end_date:
            # Filter data for the specified date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df_filtered = df[mask]
            
            # Check for significant missing data (assuming trading days)
            expected_days = 252 * 3  # Approximate trading days in 3 years
            if len(df_filtered) < expected_days * 0.95:  # Allow 5% missing days
                return None
                
            return df_filtered
        
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def calculate_increase(df):
    """
    Calculate the percentage increase in stock price over the period.
    """
    try:
        # Get first and last closing prices
        first_close = df.iloc[0]['close']
        last_close = df.iloc[-1]['close']
        
        # Calculate percentage increase
        increase = ((last_close - first_close) / first_close) * 100
        
        # Calculate additional metrics for quality check
        avg_volume = df['volume'].mean()
        avg_trades = df['trades'].mean()
        
        return {
            'increase': increase,
            'avg_volume': avg_volume,
            'avg_trades': avg_trades
        }
    except Exception as e:
        print(f"Error calculating increase: {str(e)}")
        return None

def select_stocks_normal_distribution(stock_data, total_count=200, std_range=2.0):
    """
    Select stocks based on normal distribution of returns.
    
    Args:
        stock_data: List of stock dictionaries with metrics
        total_count: Number of stocks to select
        std_range: Range of standard deviations to consider for selection
        
    Returns:
        List of selected stock dictionaries
    """
    # Extract increase percentages
    increases = np.array([stock['increase'] for stock in stock_data])
    
    # Calculate mean and standard deviation
    mean_increase = np.mean(increases)
    std_increase = np.std(increases)
    
    print(f"All stocks - Mean return: {mean_increase:.2f}%, Standard deviation: {std_increase:.2f}%")
    
    # Define the target distribution
    # Create bins across the normal distribution range
    lower_bound = mean_increase - std_range * std_increase
    upper_bound = mean_increase + std_range * std_increase
    
    # Create bins for the normal distribution
    num_bins = 10  # Number of bins to divide the distribution
    bin_edges = np.linspace(lower_bound, upper_bound, num_bins + 1)
    
    # Calculate the ideal number of stocks per bin based on normal distribution
    bin_probabilities = []
    for i in range(len(bin_edges) - 1):
        # Calculate probability mass in this bin
        prob_mass = stats.norm.cdf(bin_edges[i+1], mean_increase, std_increase) - \
                    stats.norm.cdf(bin_edges[i], mean_increase, std_increase)
        bin_probabilities.append(prob_mass)
    
    # Convert probabilities to target counts
    target_counts = [int(p * total_count) for p in bin_probabilities]
    
    # Adjust to ensure we select exactly total_count stocks
    remainder = total_count - sum(target_counts)
    for i in range(remainder):
        target_counts[i % len(target_counts)] += 1
    
    # Assign stocks to bins
    bins = [[] for _ in range(num_bins)]
    for stock in stock_data:
        for i in range(num_bins):
            if bin_edges[i] <= stock['increase'] < bin_edges[i+1]:
                bins[i].append(stock)
                break
        # Handle edge case for the upper bound
        if stock['increase'] >= bin_edges[-1]:
            bins[-1].append(stock)
    
    # Select stocks from each bin according to target counts
    selected_stocks = []
    for i, (bin_stocks, target) in enumerate(zip(bins, target_counts)):
        # Sort stocks within each bin by volume or other metrics for quality
        bin_stocks.sort(key=lambda x: x['avg_volume'], reverse=True)
        
        # Select the target number of stocks from this bin (or all if fewer available)
        selected_count = min(len(bin_stocks), target)
        selected_stocks.extend(bin_stocks[:selected_count])
        
        bin_range = f"{bin_edges[i]:.2f}% to {bin_edges[i+1]:.2f}%"
        print(f"Bin {i+1} ({bin_range}): Selected {selected_count}/{target} stocks (available: {len(bin_stocks)})")
    
    # If we couldn't fill some bins, select additional stocks from other bins
    if len(selected_stocks) < total_count:
        print(f"Selected only {len(selected_stocks)}/{total_count} stocks from distribution bins")
        
        # Create a list of all remaining stocks
        remaining_stocks = []
        for i, bin_stocks in enumerate(bins):
            selected_count = min(len(bin_stocks), target_counts[i])
            remaining_stocks.extend(bin_stocks[selected_count:])
        
        # Sort remaining by quality metrics
        remaining_stocks.sort(key=lambda x: x['avg_volume'], reverse=True)
        
        # Add enough to reach target count
        additional_needed = total_count - len(selected_stocks)
        additional_stocks = remaining_stocks[:additional_needed]
        
        print(f"Adding {len(additional_stocks)} additional stocks to reach target count")
        selected_stocks.extend(additional_stocks)
    
    print(f"\nFinal selection: {len(selected_stocks)} stocks")
    
    return selected_stocks

def main():
    # Set date range
    start_date = datetime(2020, 6, 1)
    end_date = datetime(2023, 6, 1)
    
    # Define directories
    base_dir = "/home/tourist/neu/QuantitativeTrading/data"
    high_price_dir = os.path.join(base_dir, "high_price_stocks")
    low_price_dir = os.path.join(base_dir, "low_price_stocks")
    output_dir = os.path.join(base_dir, "dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all stock files
    stock_data = []
    
    # Process both directories
    for directory in [high_price_dir, low_price_dir]:
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                ticker = filename.replace('.csv', '')
                file_path = os.path.join(directory, filename)
                
                print(f"Processing {ticker}...")
                
                df = load_and_validate_stock(file_path, start_date, end_date)
                if df is not None:
                    metrics = calculate_increase(df)
                    if metrics is not None:
                        stock_data.append({
                            'ticker': ticker,
                            'increase': metrics['increase'],
                            'avg_volume': metrics['avg_volume'],
                            'avg_trades': metrics['avg_trades'],
                            'file_path': file_path
                        })
    
    # Select stocks based on normal distribution
    print(f"\nFound {len(stock_data)} valid stocks")
    selected_stocks = select_stocks_normal_distribution(stock_data, 200)
    
    # Copy selected stock files to dataset directory
    for stock in selected_stocks:
        source_path = stock['file_path']
        dest_path = os.path.join(output_dir, f"{stock['ticker']}.csv")
        shutil.copy2(source_path, dest_path)
        
    # Save summary of selected stocks
    summary_df = pd.DataFrame(selected_stocks)
    summary_df = summary_df.drop('file_path', axis=1)  # Remove file path from summary
    summary_df.to_csv(os.path.join(output_dir, 'selected_stocks_summary.csv'), index=False)
    
    # Calculate statistics for selected stocks
    increases = [stock['increase'] for stock in selected_stocks]
    avg_increase = np.mean(increases)
    median_increase = np.median(increases)
    std_increase = np.std(increases)
    min_increase = np.min(increases)
    max_increase = np.max(increases)
    
    # Print distribution statistics for selected stocks
    print(f"\nSelected stocks return distribution statistics:")
    print(f"Mean return: {avg_increase:.2f}%")
    print(f"Median return: {median_increase:.2f}%")
    print(f"Return standard deviation: {std_increase:.2f}%")
    print(f"Minimum return: {min_increase:.2f}%")
    print(f"Maximum return: {max_increase:.2f}%")
    
    # Print top performing stocks
    print(f"\nTop 10 performing stocks in selection:")
    top_performers = sorted(selected_stocks, key=lambda x: x['increase'], reverse=True)[:10]
    for stock in top_performers:
        print(f"{stock['ticker']}: {stock['increase']:.2f}% increase, "
              f"Avg Volume: {stock['avg_volume']:,.0f}, "
              f"Avg Trades: {stock['avg_trades']:,.0f}")
              
    # Print worst performing stocks
    print(f"\nBottom 10 performing stocks in selection:")
    bottom_performers = sorted(selected_stocks, key=lambda x: x['increase'])[:10]
    for stock in bottom_performers:
        print(f"{stock['ticker']}: {stock['increase']:.2f}% increase, "
              f"Avg Volume: {stock['avg_volume']:,.0f}, "
              f"Avg Trades: {stock['avg_trades']:,.0f}")

if __name__ == "__main__":
    main()