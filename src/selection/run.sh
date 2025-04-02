# run.sh
#!/bin/bash

# Enhanced parameters
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/mockdata/stocks"
TRAIN_START="2020-02-01"
TRAIN_END="2020-9-30"
TEST_START="2020-10-01"
TEST_END="2020-12-30"
SEQ_LEN=20
BATCH_SIZE=32
EPOCHS=2000
LEARNING_RATE=0.002
D_MODEL=128
NUM_HEADS=8
NUM_LAYERS=16
DROPOUT=0.1

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Run training script with enhanced parameters
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
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT