import argparse
import torch
from torch.utils.data import DataLoader
import glob
import os
from model import StockDataset, EnhancedStockTransformer, train_model, plot_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Prediction with Transformer')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing stock CSV files')
    parser.add_argument('--train_start', type=str, required=True,
                      help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, required=True,
                      help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, required=True,
                      help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, required=True,
                      help='Testing end date (YYYY-MM-DD)')
    parser.add_argument('--seq_len', type=int, default=10,
                      help='Sequence length for prediction')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--d_model', type=int, default=64,
                      help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check data directory and files
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
    
    csv_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")
    
    # Get stock names from CSV files
    stock_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    print(f"Found {len(stock_names)} stocks: {', '.join(stock_names)}")
    
    try:
        # Create training dataset
        print("Creating training dataset...")
        train_dataset = StockDataset(
            args.data_dir, 
            args.train_start, 
            args.train_end, 
            args.seq_len
        )
        print(f"Training dataset size: {len(train_dataset)}")
        
        # Create validation dataset (using last 20% of training period)
        train_size = int(0.8 * len(train_dataset))
        train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
        val_subset = torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))
        
        # Create testing dataset
        print("Creating testing dataset...")
        test_dataset = StockDataset(
            args.data_dir, 
            args.test_start, 
            args.test_end, 
            args.seq_len
        )
        print(f"Testing dataset size: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        num_stocks = len(stock_names)
        model = EnhancedStockTransformer(
            seq_len=args.seq_len,
            num_stocks=num_stocks,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        print(f"Model created with:")
        print(f"- Sequence length: {args.seq_len}")
        print(f"- Number of stocks: {num_stocks}")
        print(f"- Model dimension: {args.d_model}")
        print(f"- Number of attention heads: {args.nhead}")
        print(f"- Number of transformer layers: {args.num_layers}")
        
        # Train model
        print("\nStarting training...")
        model = train_model(
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            device
        )
        
        # Generate predictions and plots
        print("\nGenerating predictions and performance plots...")
        plot_predictions(model, test_loader, stock_names, device)
        
        print("\nTraining completed successfully!")
        print("Generated files:")
        print("- individual_stock_comparisons.png")
        print("- actual_returns_comparison.png")
        print("- predicted_returns_comparison.png")
        print("- best_model.pth")
        
    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()