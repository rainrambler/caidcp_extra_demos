import torch
import torch.nn as nn
from typing import Optional

class DomainLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        # Use last hidden state(s)
        if self.lstm.bidirectional:
            h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_cat = h[-1]
        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat)
        return logits.squeeze(-1)

    def forward_with_states(self, x):
        """Return logits plus intermediate states for explanation.
        Returns dict with:
          embeddings: (B, T, E)
          lstm_out: (B, T, H*(2 if bi else 1))
          final_hidden: (B, H*(2 if bi else 1)) before FC
          logits: (B,)
        """
        emb = self.embedding(x)
        lstm_out, (h, c) = self.lstm(emb)
        if self.lstm.bidirectional:
            h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_cat = h[-1]
        h_drop = self.dropout(h_cat)
        logits = self.fc(h_drop).squeeze(-1)
        return {
            'embeddings': emb,
            'lstm_out': lstm_out,
            'final_hidden': h_cat,
            'logits': logits
        }
