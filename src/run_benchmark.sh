#!/bin/bash

# Define default parameters
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/dataset"
START_DATE="2020-09-01"
END_DATE="2020-12-30"
INITIAL_CAPITAL=1000000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --start_date)
      START_DATE="$2"
      shift 2
      ;;
    --end_date)
      END_DATE="$2"
      shift 2
      ;;
    --initial_capital)
      INITIAL_CAPITAL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Display run information
echo "Running benchmark with the following parameters:"
echo "Data directory: $DATA_DIR"
echo "Date range: $START_DATE to $END_DATE"
echo "Initial capital: $INITIAL_CAPITAL"
echo ""

# Run the benchmark script
python benchmark.py \
  --data_dir "$DATA_DIR" \
  --start_date "$START_DATE" \
  --end_date "$END_DATE" \
  --initial_capital "$INITIAL_CAPITAL"

# Check if the benchmark completed successfully
if [ $? -eq 0 ]; then
  echo ""
  echo "Benchmark completed successfully!"
  echo "Results have been saved to benchmark_results.csv"
  echo "Visualizations have been saved to benchmark_portfolio_value.png"
else
  echo ""
  echo "Benchmark failed with error code $?"
fi