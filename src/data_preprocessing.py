import pandas as pd
import os
from datetime import datetime
import shutil

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
    
    # Sort stocks by increase and select top 200
    stock_data.sort(key=lambda x: x['increase'], reverse=True)
    selected_stocks = stock_data[:200]
    
    # Copy selected stock files to dataset directory
    for stock in selected_stocks:
        source_path = stock['file_path']
        dest_path = os.path.join(output_dir, f"{stock['ticker']}.csv")
        shutil.copy2(source_path, dest_path)
        
    # Save summary of selected stocks
    summary_df = pd.DataFrame(selected_stocks)
    summary_df = summary_df.drop('file_path', axis=1)  # Remove file path from summary
    summary_df.to_csv(os.path.join(output_dir, 'selected_stocks_summary.csv'), index=False)
    
    print(f"\nSelected {len(selected_stocks)} stocks")
    print(f"\nTop 10 stocks by price increase:")
    for stock in selected_stocks[:10]:
        print(f"{stock['ticker']}: {stock['increase']:.2f}% increase, "
              f"Avg Volume: {stock['avg_volume']:,.0f}, "
              f"Avg Trades: {stock['avg_trades']:,.0f}")

if __name__ == "__main__":
    main()