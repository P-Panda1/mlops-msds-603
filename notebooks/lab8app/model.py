import torch
import torch.nn as nn

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_term)  # even indices
        pe[0, :, 1::2] = torch.cos(pos * div_term)  # odd indices

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=128,
                 num_heads=4,
                 num_encoder_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 num_classes=2,
                 max_seq_len=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        self.positional_encoding = PositionalEncoding(
            embed_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # important for easier batching
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        # Add initialization for transformer layers
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = self.positional_encoding(embedded)

        encoded = self.transformer_encoder(
            embedded, src_key_padding_mask=src_key_padding_mask)

        # Add NaN check for encoder output
        if torch.isnan(encoded).any():
            raise ValueError("NaN detected in encoder output")

        # Handle mean pooling with padding mask
        if src_key_padding_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            # Invert mask: True for actual tokens, False for padding
            mask = ~src_key_padding_mask  # (batch_size, seq_len)

            # Expand mask to match encoded dimensions
            mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)

            # Sum only non-padding elements
            sum_pooled = (encoded * mask_expanded).sum(dim=1)

            # Count non-padding elements per sequence, avoid division by zero
            counts = mask.sum(dim=1).unsqueeze(-1)  # (batch_size, 1)
            counts = counts.clamp(min=1.0)  # Replace 0 with 1 to avoid NaN

            pooled = sum_pooled / counts

        # Inside the forward method after pooling:
        if torch.isnan(pooled).any():
            raise ValueError("NaN detected in pooled output")

        return self.classifier(pooled)
