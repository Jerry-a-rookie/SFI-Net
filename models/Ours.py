"""Channel-independent patch Transformer with sparse channel-patch spectral interaction.

The module deliberately keeps the temporal encoder independent across channels:
the same patch projection and Transformer blocks are applied to each channel,
with no channel mixing until the CP graph module.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Augmentation import get_augmentation


class TemporalSelfAttention(nn.Module):
    """Standard multi-head self-attention over the patch axis only."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: [batch * channels, patches, d_model]
        b, p, _ = x.shape
        qkv = self.qkv(x).reshape(b, p, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if return_attention:
            weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = torch.softmax(weights, dim=-1)
            out = torch.matmul(self.attn_dropout(weights), v)
        else:
            weights = None
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).reshape(b, p, self.d_model)
        return self.out_dropout(self.proj(out)), weights


class TemporalTransformerBlock(nn.Module):
    """Transformer encoder block in the single-scale temporal path."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = TemporalSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attn_out, weights = self.attn(x, return_attention=return_attention)
        x = self.norm1(x + self.residual_dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x, weights


class Ours(nn.Module):
    """Full Ours classifier. Input shape is [B, T, C], matching TeCh loaders."""

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        num_class: int,
        patch_len: int = 32,
        stride: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 6,
        dropout: float = 0.0,
        top_k: int = 8,
        self_loop: float = 1.0,
        soft_temperature: float = 4.0,
        augmentations=None,
    ):
        super().__init__()
        if seq_len < patch_len:
            raise ValueError("seq_len must be at least patch_len")
        self.seq_len = int(seq_len)
        self.enc_in = int(enc_in)
        self.num_class = int(num_class)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.n_patches = 1 + (self.seq_len - self.patch_len) // self.stride
        self.n_nodes = self.enc_in * self.n_patches
        self.d_model = int(d_model)
        self.top_k = min(int(top_k), self.n_nodes - 1)
        self.self_loop = float(self_loop)
        self.soft_temperature = float(soft_temperature)

        self.patch_projection = nn.Linear(self.patch_len, self.d_model)
        self.patch_dropout = nn.Dropout(dropout)
        self.register_buffer("position_embedding", self._sinusoidal_position(self.n_patches, self.d_model))
        self.temporal_encoder = nn.ModuleList(
            [TemporalTransformerBlock(self.d_model, n_heads, dropout) for _ in range(e_layers)]
        )

        # Three frequency-specific graph interactions share the same sparse graph.
        self.band_projection = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for _ in range(3)]
        )
        self.gate = nn.Sequential(
            nn.Linear(self.d_model + 3, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 3),
        )
        self.classifier = nn.Linear(self.d_model, self.num_class)

        # Ordered learnable mode-index boundaries; initialized near 1/3 and 2/3.
        tau1_target = 1.0 + (self.n_nodes - 1.0) / 3.0
        tau2_target = 1.0 + 2.0 * (self.n_nodes - 1.0) / 3.0
        p1 = (tau1_target - 1.0) / (self.n_nodes - 1.0)
        p2 = (tau2_target - tau1_target) / (self.n_nodes - tau1_target)
        self.tau1_raw = nn.Parameter(torch.tensor(math.log(p1 / (1.0 - p1))))
        self.tau2_raw = nn.Parameter(torch.tensor(math.log(p2 / (1.0 - p2))))
        # Fixed-size buffers keep state_dict loadable before/after calibration.
        self.register_buffer("shared_basis", torch.eye(self.n_nodes), persistent=True)
        self.register_buffer("shared_eigenvalues", torch.zeros(self.n_nodes), persistent=True)
        self.register_buffer("basis_ready", torch.tensor(False), persistent=True)

        aug_specs = list(augmentations or ["none"])
        self.augmentations = nn.ModuleList([get_augmentation(spec) for spec in aug_specs])

    @staticmethod
    def _sinusoidal_position(length: int, d_model: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def set_shared_basis(self, eigenvectors: torch.Tensor, eigenvalues: torch.Tensor):
        if eigenvectors.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"Expected basis {(self.n_nodes, self.n_nodes)}, got {tuple(eigenvectors.shape)}"
            )
        self.shared_basis = eigenvectors.detach().to(
            device=self.patch_projection.weight.device, dtype=self.patch_projection.weight.dtype
        )
        self.shared_eigenvalues = eigenvalues.detach().to(
            device=self.patch_projection.weight.device, dtype=self.patch_projection.weight.dtype
        )
        self.basis_ready.fill_(True)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or len(self.augmentations) == 0:
            return x
        # TeCh augmentations include in-place operators; clone protects loader tensors.
        x = x.clone()
        aug = self.augmentations[random.randrange(len(self.augmentations))]
        return aug(x)

    def encode_nodes(self, x: torch.Tensor, return_attention: bool = False):
        """Return CP node tokens [B, C*P, D] and optional [L,B,C,H,P,P] maps."""
        if x.ndim != 3:
            raise ValueError(f"Expected input [B,T,C], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len or x.shape[2] != self.enc_in:
            raise ValueError(
                f"Expected [B,{self.seq_len},{self.enc_in}], got {tuple(x.shape)}"
            )
        x = self._augment(x).transpose(1, 2).contiguous()  # [B,C,T]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        b, c, p, _ = patches.shape
        tokens = patches.reshape(b * c, p, self.patch_len)
        tokens = self.patch_dropout(self.patch_projection(tokens) + self.position_embedding)

        attention_maps = []
        for layer in self.temporal_encoder:
            tokens, weights = layer(tokens, return_attention=return_attention)
            if return_attention:
                attention_maps.append(
                    weights.reshape(b, c, weights.shape[1], p, p).detach()
                )
        tokens = tokens.reshape(b, c, p, self.d_model)
        nodes = tokens.reshape(b, self.n_nodes, self.d_model)
        if return_attention:
            return nodes, torch.stack(attention_maps, dim=0)
        return nodes, None

    def build_graph(self, nodes: torch.Tensor):
        """Cosine Top-k graph, symmetrized with self-loops; all samples are separate graphs."""
        normalized = F.normalize(nodes, p=2, dim=-1, eps=1e-8)
        sim = torch.bmm(normalized, normalized.transpose(1, 2)).clamp_min(0.0)
        n = sim.shape[-1]
        diagonal = torch.eye(n, dtype=torch.bool, device=sim.device).unsqueeze(0)
        sim = sim.masked_fill(diagonal, 0.0)
        top_values, top_indices = torch.topk(sim, k=self.top_k, dim=-1, largest=True, sorted=False)
        directed = torch.zeros_like(sim).scatter(-1, top_indices, top_values)
        adjacency = 0.5 * (directed + directed.transpose(1, 2))
        adjacency = adjacency + self.self_loop * torch.eye(n, device=sim.device, dtype=sim.dtype)
        degree = adjacency.sum(dim=-1).clamp_min(1e-8)
        inv_sqrt_degree = degree.rsqrt()
        normalized_adjacency = (
            inv_sqrt_degree.unsqueeze(-1)
            * adjacency
            * inv_sqrt_degree.unsqueeze(-2)
        )
        laplacian = torch.eye(n, device=sim.device, dtype=sim.dtype).unsqueeze(0) - normalized_adjacency
        return adjacency, normalized_adjacency, laplacian

    def _soft_bands(self, device, dtype):
        n = self.n_nodes
        tau1 = 1.0 + (n - 1.0) * torch.sigmoid(self.tau1_raw)
        tau2 = tau1 + (n - tau1) * torch.sigmoid(self.tau2_raw)
        modes = torch.arange(1, n + 1, device=device, dtype=dtype)
        alpha_low = torch.sigmoid(self.soft_temperature * (tau1 - modes))
        alpha_high = torch.sigmoid(self.soft_temperature * (modes - tau2))
        alpha_mid = (1.0 - alpha_low - alpha_high).clamp_min(0.0)
        alpha = torch.stack([alpha_low, alpha_mid, alpha_high], dim=-1)
        # Numerical normalization preserves a partition of unity.
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return alpha, tau1, tau2

    def forward(self, x: torch.Tensor, return_details: bool = False, return_embedding: bool = False):
        nodes, attention = self.encode_nodes(x, return_attention=return_details)
        adjacency, normalized_adjacency, laplacian = self.build_graph(nodes)
        if not bool(self.basis_ready):
            raise RuntimeError("Shared spectral basis is not initialized; calibrate it on train data first.")
        basis = self.shared_basis.to(device=nodes.device, dtype=nodes.dtype)

        alpha, tau1, tau2 = self._soft_bands(nodes.device, nodes.dtype)
        spectral_nodes = torch.matmul(basis.transpose(0, 1), nodes)
        band_nodes = []
        for band in range(3):
            filtered = spectral_nodes * alpha[:, band].view(1, -1, 1)
            z_band = torch.matmul(basis, filtered)
            interacted = torch.bmm(normalized_adjacency, z_band)
            band_nodes.append(F.gelu(self.band_projection[band](interacted)))

        spectral_energy = spectral_nodes.square().sum(dim=-1)
        node_band_energy = torch.matmul(
            basis.square(),
            spectral_energy.unsqueeze(-1) * alpha.unsqueeze(0),
        )
        gate_logits = self.gate(torch.cat([nodes, node_band_energy], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        fused = sum(gate[..., band : band + 1] * band_nodes[band] for band in range(3))
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled)

        if return_embedding and not return_details:
            # Export classifier-input embeddings without materializing attention maps.
            return logits, pooled
        if not return_details:
            return logits
        details = {
            "pooled": pooled.detach(),
            "nodes": nodes.detach(),
            "adjacency": adjacency.detach(),
            "normalized_adjacency": normalized_adjacency.detach(),
            "laplacian": laplacian.detach(),
            "attention": attention,
            "alpha": alpha.detach(),
            "tau1": tau1.detach(),
            "tau2": tau2.detach(),
            "gate": gate.detach(),
            "node_band_energy": node_band_energy.detach(),
            "eigenvalues": self.shared_eigenvalues.detach(),
        }
        return logits, details


def build_model(config):
    """Small factory for the dedicated runner and external scripts."""
    return Ours(**config)
