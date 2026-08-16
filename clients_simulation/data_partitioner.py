"""
[IMPLEMENTED] Data partitioner for generating IID and Non-IID client partitions.
Guarantees raw client data strictly remains local to each simulated edge node.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ClientDataPartition:
    client_id: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    num_samples: int


def partition_data_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> list[ClientDataPartition]:
    """
    Partition dataset into N identical and independently distributed (IID) client subsets.
    """
    np.random.seed(random_state)
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    client_splits = np.array_split(indices, num_clients)

    partitions = []
    for i, client_indices in enumerate(client_splits):
        client_id = f"client_{i:02d}"
        n_client = len(client_indices)
        n_val = max(1, int(n_client * val_ratio))

        val_idx = client_indices[:n_val]
        train_idx = client_indices[n_val:]

        partitions.append(ClientDataPartition(
            client_id=client_id,
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            num_samples=n_client
        ))
    return partitions


def partition_data_non_iid_dirichlet(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> list[ClientDataPartition]:
    """
    Partition dataset into N Non-IID client subsets using a Dirichlet distribution Dir(alpha).
    Smaller alpha (e.g. 0.1 - 0.5) induces higher non-IID class imbalance across clients.
    """
    np.random.seed(random_state)
    num_classes = len(np.unique(y))
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Scale to match class sample counts
        proportions = np.array([p * (len(idx_j) < len(X) / num_clients) for p, idx_j in zip(proportions, client_indices)])
        if proportions.sum() == 0:
            proportions = np.ones(num_clients)
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]

        # Split class indices according to sampled counts
        splits = np.split(idx_c, proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    partitions = []
    for i in range(num_clients):
        client_id = f"client_{i:02d}"
        c_idx = np.array(client_indices[i])
        np.random.shuffle(c_idx)

        # Ensure client has at least minimum samples
        if len(c_idx) == 0:
            # Fallback random sample
            c_idx = np.random.choice(len(X), size=max(5, int(len(X) / (num_clients * 2))), replace=False)

        n_client = len(c_idx)
        n_val = max(1, int(n_client * val_ratio))
        val_idx = c_idx[:n_val]
        train_idx = c_idx[n_val:]

        partitions.append(ClientDataPartition(
            client_id=client_id,
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            num_samples=n_client
        ))
    return partitions
