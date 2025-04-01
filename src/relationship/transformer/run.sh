#!/bin/bash

# 使用说明
function usage {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -d, --data_dir DIR      股票CSV文件所在目录（必需）"
    echo "  -s, --start_date DATE   开始日期，格式：YYYY-MM-DD（必需）"
    echo "  -e, --end_date DATE     结束日期，格式：YYYY-MM-DD（必需）"
    echo "  -o, --output_dir DIR    输出目录（默认：./output）"
    echo "  -b, --batch_size SIZE   批次大小（默认：32）"
    echo "  -ep, --epochs NUM       训练轮数（默认：100）"
    echo "  -lr, --learning_rate LR 学习率（默认：0.001）"
    echo "  -sl, --seq_len LEN      序列长度（默认：10）"
    echo "  -dm, --d_model DIM      模型维度（默认：64）"
    echo "  -nh, --nhead NUM        多头注意力头数（默认：8）"
    echo "  -nl, --num_layers NUM   Transformer层数（默认：2）"
    echo "  -dr, --dropout RATE     Dropout率（默认：0.1）"
    echo "  -dv, --device DEV       使用的设备（默认：cuda如果可用，否则cpu）"
    echo "  -h, --help              显示此帮助信息"
    exit 1
}

# 默认参数
DATA_DIR="/home/tourist/neu/QuantitativeTrading/data/mockdata/stocks"
START_DATE="2020-01-01"
END_DATE="2020-02-20"
OUTPUT_DIR="./output"
BATCH_SIZE=32
EPOCHS=1000
LEARNING_RATE=0.001
SEQ_LEN=10
D_MODEL=64
NHEAD=8
NUM_LAYERS=2
DROPOUT=0.1
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -s|--start_date)
            START_DATE="$2"
            shift 2
            ;;
        -e|--end_date)
            END_DATE="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -ep|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -lr|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -sl|--seq_len)
            SEQ_LEN="$2"
            shift 2
            ;;
        -dm|--d_model)
            D_MODEL="$2"
            shift 2
            ;;
        -nh|--nhead)
            NHEAD="$2"
            shift 2
            ;;
        -nl|--num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        -dr|--dropout)
            DROPOUT="$2"
            shift 2
            ;;
        -dv|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

# 检查必需参数
if [ -z "$DATA_DIR" ] || [ -z "$START_DATE" ] || [ -z "$END_DATE" ]; then
    echo "错误: 必须提供数据目录、开始日期和结束日期。"
    usage
fi

# 检查日期格式
date_regex="^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
if ! [[ $START_DATE =~ $date_regex ]] || ! [[ $END_DATE =~ $date_regex ]]; then
    echo "错误: 日期格式必须为 YYYY-MM-DD。"
    usage
fi

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录 '$DATA_DIR' 不存在。"
    exit 1
fi

# 检查是否有CSV文件
csv_count=$(find "$DATA_DIR" -name "*.csv" | wc -l)
if [ "$csv_count" -eq 0 ]; then
    echo "错误: 数据目录 '$DATA_DIR' 中没有找到CSV文件。"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置设备参数
if [ -z "$DEVICE" ]; then
    # 检查是否有CUDA
    if command -v nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

echo "====== 股票影响关系分析 ======"
echo "数据目录: $DATA_DIR"
echo "时间范围: $START_DATE 至 $END_DATE"
echo "输出目录: $OUTPUT_DIR"
echo "使用设备: $DEVICE"
echo "============================="

# 运行Python脚本
python stock_transformer.py \
    --data_dir "$DATA_DIR" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --seq_len "$SEQ_LEN" \
    --d_model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num_layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --device "$DEVICE"

# 检查脚本是否成功运行
if [ $? -eq 0 ]; then
    echo "分析完成！结果保存在: $OUTPUT_DIR"
    echo "关系矩阵已保存为: $OUTPUT_DIR/relation_matrix.csv"
    echo "关系矩阵可视化: $OUTPUT_DIR/relation_matrix.png"
else
    echo "分析过程中出现错误，请检查日志文件。"
fi