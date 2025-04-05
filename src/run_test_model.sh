# run.sh
#!/bin/bash

# Enhanced parameters
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/dataset"
TRAIN_START="2020-06-01"
TRAIN_END="2021-01-01"
TEST_START="2021-01-02"
TEST_END="2021-4-01"
SEQ_LEN=25
BATCH_SIZE=32
EPOCHS=500
LEARNING_RATE=0.001
D_MODEL=128
NUM_HEADS=4
NUM_LAYERS=4
DROPOUT=0.1
EARLY_STOP=25

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Run training script with enhanced parameters
python test_model.py \
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
    --dropout $DROPOUT \
    --early_stop $EARLY_STOP