import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomEmbedding(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x: Tensor):
        target = torch.zeros(1, x.size()[1], self.d_model, device=x.device)
        target[:, :, :x.size()[2]] = x
        return target

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000, downscale: int = 1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        position = position/downscale
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len , d_model)
        pe[0,: , 0::2] = torch.sin(position * div_term)
        pe[0,: , 1::2] = torch.cos(position * div_term)
        pe = F.pad(input=pe, pad=(0 , 0, 1, 0), mode='constant', value=0)
        pe = pe[:,:max_len,:]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        test = self.pe[:,:x.size(1),:]
        x = x + test
        return self.dropout(x)
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor):
        # Self-Attention with Residual Connection
        x = x.squeeze(0)
        x = x + self.dropout(self.self_attn(x, x, x)[0])
        x = self.norm1(x)
        
        # Feedforward with Residual Connection
        x = x + self.dropout(self.ffn(x))
        x = self.norm2(x)
        x = x.unsqueeze(0)

        return x

class CustomEncoder(nn.Module):
    def __init__(self,d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float=0.1, max_len: int=1750, downscale: int = 1):
        super(CustomEncoder, self).__init__()
        self.embedding = CustomEmbedding(d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len, downscale=downscale)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
    
    def forward(self, x):
        x = self.positional_encoding(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        return x