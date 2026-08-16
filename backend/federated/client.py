"""
[IMPLEMENTED] Flower Client wrapper and classical PyTorch tabular neural network.
Training executes strictly on classical local client hardware.
"""
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.models.transformer_model import TimeSeriesTransformerNN
from backend.security.dp_mechanism import add_gaussian_noise, clip_parameter_tensors
from clients_simulation.data_partitioner import ClientDataPartition


class TabularPyTorchNN(nn.Module):
    """Classical Tabular Neural Network trained on edge client hardware."""

    def __init__(self, in_features: int = 30, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract PyTorch parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set PyTorch model state_dict from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters, strict=True)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowerTabularClient(fl.client.NumPyClient):
    """Flower NumPyClient implementation supporting both Tabular and Transformer sequence architectures."""

    def __init__(
        self,
        client_id: str,
        partition: ClientDataPartition,
        in_features: int = 30,
        hidden_dim: int = 64,
        model_architecture: str = "tabular_nn",
        device: str | None = None
    ):
        self.client_id = client_id
        self.partition = partition
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_architecture = model_architecture

        if model_architecture == "transformer":
            self.model = TimeSeriesTransformerNN(
                in_features=in_features,
                d_model=32,
                nhead=4,
                num_layers=2,
                num_classes=2
            ).to(self.device)
        else:
            self.model = TabularPyTorchNN(in_features=in_features, hidden_dim=hidden_dim).to(self.device)


    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        # Update local model with global parameters
        set_model_parameters(self.model, parameters)

        epochs = int(config.get("local_epochs", 2))
        batch_size = int(config.get("batch_size", 32))
        lr = float(config.get("learning_rate", 0.01))

        # Build DataLoader for client's private partition
        X_tensor = torch.tensor(self.partition.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.partition.y_train, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

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
                total_samples += len(batch_y)
        
        updated_params = get_model_parameters(self.model)

        # Optional Differential Privacy: L2 clipping + calibrated Gaussian perturbation
        dp_enabled = bool(config.get("dp_enabled", False))
        if dp_enabled:
            clip_norm = float(config.get("dp_clip_norm", 1.0))
            epsilon = float(config.get("dp_epsilon", 1.0))
            delta = float(config.get("dp_delta", 1e-5))
            clipped_params, _ = clip_parameter_tensors(updated_params, clip_norm=clip_norm)
            updated_params, _ = add_gaussian_noise(clipped_params, clip_norm=clip_norm, epsilon=epsilon, delta=delta)

        avg_loss = total_loss / max(1, total_samples)
        train_acc = correct / max(1, total_samples)

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(avg_loss),
            "train_accuracy": float(train_acc),
            "dp_enabled": dp_enabled
        }
        return updated_params, len(self.partition.X_train), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        set_model_parameters(self.model, parameters)
        self.model.eval()

        X_val = torch.tensor(self.partition.X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(self.partition.y_val, dtype=torch.long).to(self.device)

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = self.model(X_val)
            val_loss = criterion(outputs, y_val).item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = y_val.cpu().numpy()

        acc = float(accuracy_score(targets, preds))
        f1 = float(f1_score(targets, preds, zero_division=0))

        metrics = {
            "val_accuracy": acc,
            "val_f1": f1
        }
        return float(val_loss), len(self.partition.X_val), metrics
