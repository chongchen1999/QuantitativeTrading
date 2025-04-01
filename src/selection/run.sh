#!/bin/bash

# Default parameters
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/mockdata/stocks"
TRAIN_START="2020-02-01"
TRAIN_END="2021-02-28"
TEST_START="2021-03-01"
TEST_END="2021-07-01"
SEQ_LEN=20
BATCH_SIZE=32
EPOCHS=500
LEARNING_RATE=0.001
D_MODEL=64
NHEAD=8
NUM_LAYERS=6
DROPOUT=0.1

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Run training script
python main.py \
    --data_dir $DATA_DIR \
    --train_start $TRAIN_START \
    --train_end $TRAIN_END \
    --test_start $TEST_START \
    --test_end $TEST_END \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --d_model $D_MODEL \
    --nhead $NHEAD \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT