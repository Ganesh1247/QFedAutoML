"""
[IMPLEMENTED] Classical PyTorch Transformer architecture for sequential / sensor client data.
Note: Neural network training ALWAYS runs on classical hardware (PyTorch, CPU/GPU).
"""
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.models.base_model import BaseModelWrapper


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformerNN(nn.Module):
    """Classical PyTorch Multi-Head Self-Attention Transformer for edge sequence processing."""

    def __init__(
        self,
        in_features: int = 6,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model

        # Project raw sensor features into embedding dimension
        self.input_projection = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 2D inputs (batch_size, in_features) by unsqueezing to (batch_size, 1, in_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Project and encode
        x_emb = self.input_projection(x) * math.sqrt(self.d_model)
        x_pos = self.pos_encoder(x_emb)
        encoded = self.transformer_encoder(x_pos)

        # Global average pooling over time steps
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class TransformerModelWrapper(BaseModelWrapper):
    """Wrapper exposing standard fit / predict interface for TimeSeriesTransformerNN."""

    def __init__(
        self,
        in_features: int = 6,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        lr: float = 0.005,
        epochs: int = 10,
        batch_size: int = 32,
        device: str | None = None
    ):
        self.in_features = in_features
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = TimeSeriesTransformerNN(
            in_features=in_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict[str, float]:
        epochs = kwargs.get("epochs", self.epochs)
        batch_size = kwargs.get("batch_size", self.batch_size)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

        self.is_fitted = True
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return {"train_loss": float(avg_loss), "train_accuracy": float(acc)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def get_weights(self) -> list[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), weights, strict=True)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.is_fitted = True
