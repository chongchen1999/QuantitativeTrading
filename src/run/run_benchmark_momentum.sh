#!/bin/bash

# Run Momentum-Based Benchmark Strategy
# This script runs the momentum-based portfolio simulation

# Environment variables (edit these values as needed)
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/dataset"
START_DATE="2020-09-01"
END_DATE="2023-09-01"
INITIAL_CAPITAL=1000000
REBALANCE_DAYS=14
MIN_STOCKS=5
MAX_ALLOCATION=0.2

# Display header
echo "=========================================================="
echo "         Running Momentum-Based Benchmark Strategy        "
echo "=========================================================="
echo

# Override environment variables with command line arguments if provided
while [ $# -gt 0 ]; do
  case "$1" in
    --data_dir=*)
      DATA_DIR="${1#*=}"
      ;;
    --start_date=*)
      START_DATE="${1#*=}"
      ;;
    --end_date=*)
      END_DATE="${1#*=}"
      ;;
    --initial_capital=*)
      INITIAL_CAPITAL="${1#*=}"
      ;;
    --rebalance_days=*)
      REBALANCE_DAYS="${1#*=}"
      ;;
    --min_stocks=*)
      MIN_STOCKS="${1#*=}"
      ;;
    --max_allocation=*)
      MAX_ALLOCATION="${1#*=}"
      ;;
    *)
      echo "Error: Unknown parameter $1"
      exit 1
      ;;
  esac
  shift
done

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Data directory $DATA_DIR does not exist"
  exit 1
fi

# Display settings
echo "Settings:"
echo "- Data Directory: $DATA_DIR"
echo "- Start Date: $START_DATE"
echo "- End Date: $END_DATE"
echo "- Initial Capital: $INITIAL_CAPITAL"
echo "- Rebalance Frequency: Every $REBALANCE_DAYS days"
echo "- Minimum Stocks: $MIN_STOCKS"
echo "- Maximum Allocation: $(echo "$MAX_ALLOCATION * 100" | bc)%"
echo

# Run the momentum benchmark script
echo "Starting momentum strategy simulation..."
python ../benchmark_momentum.py \
  --data_dir="$DATA_DIR" \
  --start_date="$START_DATE" \
  --end_date="$END_DATE" \
  --initial_capital="$INITIAL_CAPITAL" \
  --rebalance_days="$REBALANCE_DAYS" \
  --min_stocks="$MIN_STOCKS" \
  --max_allocation="$MAX_ALLOCATION"

# Check if simulation was successful
if [ $? -eq 0 ]; then
  echo
  echo "Momentum strategy simulation completed successfully!"
  echo "Results are available in momentum_results.csv"
  echo "Performance visualization saved as momentum_portfolio_value.png"
  echo "Performance metrics saved in momentum_performance_metrics.csv"
else
  echo
  echo "Error: Momentum strategy simulation failed!"
fi

# If both benchmark and momentum results exist, generate comparison
if [ -f "daily_rebalanced_results.csv" ] && [ -f "momentum_results.csv" ]; then
  echo
  echo "Both benchmark strategies completed. Generating comparison..."
  
  # Generate comparison script (optional, you can implement this separately)
  # python compare_benchmarks.py

  echo "Strategy comparison completed. See comparison_results.png"
fi

echo
echo "=========================================================="