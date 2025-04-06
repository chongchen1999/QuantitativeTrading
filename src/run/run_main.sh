#!/bin/bash

# Default values
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/dataset"
START_DATE="2020-09-01"
END_DATE="2023-09-01"
TRAINING_WINDOW=180
UPDATE_INTERVAL=20
SOFTMAX_THRESHOLD=0.8
MAX_STOCKS=200
INITIAL_CAPITAL=1000000

# Model parameters
SEQ_LEN=20
BATCH_SIZE=32
D_MODEL=128
NUM_HEADS=4
NUM_LAYERS=4
DROPOUT=0.1
EPOCHS=100
LEARNING_RATE=0.001

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
        --training_window)
            TRAINING_WINDOW="$2"
            shift 2
            ;;
        --update_interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --softmax_threshold)
            SOFTMAX_THRESHOLD="$2"
            shift 2
            ;;
        --max_stocks)
            MAX_STOCKS="$2"
            shift 2
            ;;
        --initial_capital)
            INITIAL_CAPITAL="$2"
            shift 2
            ;;
        --seq_len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --num_heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Run the main Python script with all parameters
python ../main.py \
    --data_dir "$DATA_DIR" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --training_window "$TRAINING_WINDOW" \
    --update_interval "$UPDATE_INTERVAL" \
    --softmax_threshold "$SOFTMAX_THRESHOLD" \
    --max_stocks "$MAX_STOCKS" \
    --initial_capital "$INITIAL_CAPITAL" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --d_model "$D_MODEL" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE"