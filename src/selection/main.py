# main.py
import argparse
from torch.utils.data import DataLoader, random_split
from model import StockDataset, StockTransformer, train_model, plot_predictions
import glob
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Prediction with Enhanced Transformer')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing stock CSV files')
    parser.add_argument('--train_start', type=str, required=True, help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, required=True, help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, required=True, help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, required=True, help='Testing end date (YYYY-MM-DD)')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length for prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
    
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")
    
    stock_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    print(f"Found {len(stock_names)} stocks: {', '.join(stock_names)}")
    
    print("Creating training dataset...")
    train_dataset = StockDataset(args.data_dir, args.train_start, args.train_end, args.seq_len)
    
    print("Creating testing dataset...")
    test_dataset = StockDataset(args.data_dir, args.test_start, args.test_end, args.seq_len)
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_stocks = len(stock_names)
    model = StockTransformer(
        seq_len=args.seq_len,
        num_stocks=num_stocks,
        num_features=5,  # returns, volume_ma5, price_ma5, price_ma20, volatility
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    print(f"Model created with sequence length {args.seq_len} for {num_stocks} stocks")
    
    print("Starting training...")
    train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    
    print("Generating predictions plot...")
    plot_predictions(model, test_loader, stock_names, device)

if __name__ == "__main__":
    main()