from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    """Per-pixel student head for dynamic uncertainty plus a 3D-aware latent."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        latent_dim: int = 3,
        net_depth: int = 2,
        net_activation=F.gelu,
        weight_init: str = "he_uniform",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.net_activation = net_activation
        self.softplus = nn.Softplus()

        self.layers = nn.ModuleList()
        for i in range(net_depth):
            dense_layer = nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            if weight_init == "he_uniform":
                nn.init.kaiming_uniform_(dense_layer.weight, nonlinearity="relu")
            elif weight_init == "xavier_uniform":
                nn.init.xavier_uniform_(dense_layer.weight)
            else:
                raise NotImplementedError(
                    f"Unknown weight initialization method {weight_init}"
                )
            nn.init.zeros_(dense_layer.bias)
            self.layers.append(dense_layer)

        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.kaiming_uniform_(self.uncertainty_head.weight, nonlinearity="relu")
        nn.init.zeros_(self.uncertainty_head.bias)
        nn.init.xavier_uniform_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)

    def _flatten_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
        if x.dim() == 3:
            if x.shape[-1] == self.input_dim:
                return x.unsqueeze(0), x.shape, False
            if x.shape[0] == self.input_dim:
                return x.permute(1, 2, 0).unsqueeze(0), x.shape, False
            raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

        if x.dim() == 4:
            if x.shape[-1] == self.input_dim:
                return x, x.shape, True
            if x.shape[1] == self.input_dim:
                return x.permute(0, 2, 3, 1).contiguous(), x.shape, True
            raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

        raise ValueError(f"Unsupported feature rank: {x.dim()}")

    def forward(self, x: torch.Tensor):
        x_hwc, original_shape, batched = self._flatten_features(x)
        b, h, w, c = x_hwc.shape
        x_flat = x_hwc.view(-1, c)

        for layer in self.layers:
            x_flat = layer(x_flat)
            x_flat = self.net_activation(x_flat)
            x_flat = F.dropout(x_flat, p=0.2, training=self.training)

        uncertainty = self.softplus(self.uncertainty_head(x_flat)) + 1e-4
        latent = torch.tanh(self.latent_head(x_flat))

        uncertainty = uncertainty.view(b, h, w)
        latent = latent.view(b, h, w, self.latent_dim)

        if not batched:
            uncertainty = uncertainty.squeeze(0)
            latent = latent.squeeze(0)

        return uncertainty, latent


def generate_uncertainty_mlp(
    n_features: int,
    latent_dim: int = 3,
    hidden_dim: int = 128,
    net_depth: int = 2,
) -> MLPNetwork:
    network = MLPNetwork(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        net_depth=net_depth,
    ).cuda()
    return network
