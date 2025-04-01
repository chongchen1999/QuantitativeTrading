import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

class StockRelationshipTransformer(nn.Module):
    def __init__(self, num_stocks, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        """
        Stock Relationship Transformer Model
        
        Args:
            num_stocks: Number of stocks (N)
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(StockRelationshipTransformer, self).__init__()
        self.num_stocks = num_stocks
        self.d_model = d_model
        
        # Embedding layer for stock returns
        self.stock_embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Relationship matrix projection
        self.relation_proj = nn.Linear(d_model, 1)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_stocks, seq_len)
               where seq_len is the time window length (n days)
        
        Returns:
            relation_matrix: Tensor of shape (batch_size, num_stocks, num_stocks)
                            with values in [-1, 1] representing influence
        """
        batch_size, num_stocks, seq_len = x.shape
        
        # Reshape and embed returns
        x = x.unsqueeze(-1)  # (batch_size, num_stocks, seq_len, 1)
        x = self.stock_embedding(x)  # (batch_size, num_stocks, seq_len, d_model)
        
        # Reshape for transformer: combine stocks and sequence as one dimension
        x = x.permute(1, 0, 2, 3)  # (num_stocks, batch_size, seq_len, d_model)
        x = x.reshape(num_stocks, batch_size * seq_len, self.d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (num_stocks, batch_size*seq_len, d_model)
        
        # Reshape back
        encoded = encoded.reshape(num_stocks, batch_size, seq_len, self.d_model)
        encoded = encoded.permute(1, 0, 2, 3)  # (batch_size, num_stocks, seq_len, d_model)
        
        # Calculate relationship matrix
        # For each stock j, we want to see how much it's influenced by other stocks i
        relation_scores = torch.zeros(batch_size, num_stocks, num_stocks, device=x.device)
        
        for j in range(num_stocks):
            # Get the next day's return for stock j (shifted by 1)
            # We use the embedding from day t to predict day t+1
            target_emb = encoded[:, j, 1:, :]  # (batch_size, seq_len-1, d_model)
            
            # Get the current day's return for all stocks
            source_embs = encoded[:, :, :-1, :]  # (batch_size, num_stocks, seq_len-1, d_model)
            
            # Calculate attention scores between source_embs and target_emb
            # Reshape for matrix multiplication
            target_emb_flat = target_emb.reshape(batch_size * (seq_len-1), self.d_model)
            source_embs_flat = source_embs.permute(0, 2, 1, 3).reshape(batch_size * (seq_len-1), num_stocks, self.d_model)
            
            # Calculate relationship scores
            scores = torch.matmul(source_embs_flat, target_emb_flat.unsqueeze(-1)).squeeze(-1)
            scores = scores.reshape(batch_size, seq_len-1, num_stocks)
            scores = scores.mean(dim=1)  # Average over time steps
            
            # Normalize scores to [-1, 1] using tanh
            scores = torch.tanh(scores)
            relation_scores[:, :, j] = scores
        
        return relation_scores


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StockDataset(Dataset):
    def __init__(self, price_data, window_size=20):
        """
        Args:
            price_data: numpy array of shape (num_stocks, num_days)
            window_size: size of the time window to use
        """
        super(StockDataset, self).__init__()
        self.num_stocks, self.num_days = price_data.shape
        self.window_size = window_size
        
        # Calculate log returns
        self.returns = np.log(price_data[:, 1:] / price_data[:, :-1])
        
        # Create sliding windows
        self.windows = []
        for i in range(self.num_days - window_size - 1):
            window = self.returns[:, i:i+window_size]
            self.windows.append(window)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])


def train_model(model, dataloader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            relation_matrix = model(batch)
            
            # Create target: we want to predict next day's return for each stock
            # based on current day's returns of all stocks
            target = batch[:, :, 1:]  # Next day returns
            current = batch[:, :, :-1]  # Current day returns
            
            # Calculate the actual influence (simplified approach)
            # This is a placeholder - in practice you might want a different loss
            # that better captures the lead-lag relationship
            actual_influence = torch.matmul(current.permute(0, 2, 1), target.permute(0, 2, 1))
            actual_influence = actual_influence.mean(dim=1)  # Average over time
            actual_influence = torch.tanh(actual_influence)  # Normalize to [-1, 1]
            
            # Calculate loss
            loss = criterion(relation_matrix, actual_influence)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def get_relationship_matrix(model, price_data, window_size=20):
    """
    Get the average relationship matrix for the given price data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    dataset = StockDataset(price_data, window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_relations = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            relation_matrix = model(batch)
            all_relations.append(relation_matrix.cpu().numpy())
    
    # Average over all batches
    avg_relation = np.mean(np.concatenate(all_relations, axis=0), axis=0)
    return avg_relation


# Example usage
if __name__ == "__main__":
    # Generate synthetic data: 10 stocks, 500 days
    np.random.seed(42)
    num_stocks = 10
    num_days = 500
    price_data = np.cumprod(1 + np.random.randn(num_stocks, num_days) * 0.01, axis=1) * 100
    
    # Create dataset and dataloader
    window_size = 20
    dataset = StockDataset(price_data, window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = StockRelationshipTransformer(num_stocks=num_stocks)
    
    # Train model
    train_model(model, dataloader, epochs=30)
    
    # Get final relationship matrix
    relation_matrix = get_relationship_matrix(model, price_data, window_size)
    print("Relationship matrix shape:", relation_matrix.shape)
    print("Sample relationships:")
    print(relation_matrix[:3, :3])  # Print first 3x3 subset