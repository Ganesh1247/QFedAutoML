"""
[IMPLEMENTED] Multi-Head Self-Attention Weight Visualizer for Transformer Models.
Extracts per-head and layer-wise attention matrices and computes temporal importance heatmaps.
"""
from typing import Any

import numpy as np
import torch

from backend.models.transformer_model import TimeSeriesTransformerNN, TransformerModelWrapper


class AttentionVisualizer:
    """Extracts and computes self-attention rollout maps for sequence transformers."""

    @classmethod
    def extract_attention_maps(
        cls,
        model: TimeSeriesTransformerNN | TransformerModelWrapper,
        sequence_sample: np.ndarray | torch.Tensor
    ) -> dict[str, Any]:
        """
        Pass a single sequence through the Transformer model and extract
        per-head self-attention weight matrices: Shape (num_heads, seq_len, seq_len).
        """
        # Unwrap PyTorch model if wrapped
        nn_module: TimeSeriesTransformerNN = model.model if isinstance(model, TransformerModelWrapper) else model
        nn_module.eval()

        if isinstance(sequence_sample, np.ndarray):
            x_tensor = torch.tensor(sequence_sample, dtype=torch.float32)
        else:
            x_tensor = sequence_sample.clone().detach().to(torch.float32)

        # Ensure shape: (batch_size=1, seq_len, in_features)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        device = next(nn_module.parameters()).device
        x_tensor = x_tensor.to(device)

        _batch_size, seq_len, _ = x_tensor.shape

        with torch.no_grad():
            # 1. Project input and apply positional encoding
            embedded = nn_module.input_projection(x_tensor)
            src = nn_module.pos_encoder(embedded)

            layer_attentions = []

            # 2. Forward through each encoder layer and extract attention weights
            for layer in nn_module.transformer_encoder.layers:
                # MultiheadAttention forward with need_weights=True
                # PyTorch batch_first: src shape is (batch, seq_len, d_model)
                _attn_out, attn_weights = layer.self_attn(
                    src, src, src,
                    need_weights=True,
                    average_attn_weights=False
                )
                # attn_weights shape: (batch, num_heads, seq_len, seq_len)
                layer_attentions.append(attn_weights.cpu().numpy()[0])
                # Residual + norm + feedforward step
                src = layer(src)

        # Use last layer attention as primary explainability representation
        last_layer_weights = layer_attentions[-1]  # shape: (num_heads, seq_len, seq_len)
        num_heads = last_layer_weights.shape[0]

        # Average attention across all attention heads
        avg_attention = np.mean(last_layer_weights, axis=0)  # (seq_len, seq_len)

        # Temporal sequence importance (sum over columns: how much other steps attend to step t)
        temporal_importance = np.mean(avg_attention, axis=0).tolist()

        return {
            "model_type": "TimeSeriesTransformer",
            "num_layers": len(layer_attentions),
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "averaged_attention_matrix": [[round(float(v), 5) for v in row] for row in avg_attention],
            "head_attention_matrices": [
                [[round(float(v), 5) for v in row] for row in last_layer_weights[h]]
                for h in range(num_heads)
            ],
            "temporal_importance": [round(float(v), 5) for v in temporal_importance],
            "top_attended_timestep": int(np.argmax(temporal_importance))
        }


attention_visualizer = AttentionVisualizer()
