from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_features: int
    n_metadata: int
    max_len: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    protocol_feature_indices: tuple[int, ...] = ()
    sde_dt: float = 0.05
    sde_noise_scale: float = 0.0


def _mask_aware_encoded_input(
    x: torch.Tensor,
    feature_mask: torch.Tensor | None,
    input_projection: nn.Linear,
    missingness_projection: nn.Linear,
) -> torch.Tensor:
    """Encode numeric values and their observation state separately.

    A numerical zero can be a measured value or a finite placeholder for a
    missing cell.  The producer-supplied mask makes the two cases distinct.
    Legacy callers omit the mask and therefore retain all-observed behavior.
    """
    observed = torch.ones_like(x) if feature_mask is None else feature_mask.to(dtype=x.dtype)
    if observed.shape != x.shape:
        raise ValueError("feature_mask must have the same [batch, time, channel] shape as x")
    return input_projection(x * observed) + missingness_projection(observed)


def _phase_context(
    x: torch.Tensor, phase: torch.Tensor | None,
    phase_mask: torch.Tensor | None, embedding: nn.Embedding,
) -> torch.Tensor:
    codes = phase.clamp(0, 3).long() if phase is not None else torch.zeros_like(x[..., 0], dtype=torch.long)
    if phase_mask is None:
        return embedding(codes)
    if phase_mask.shape != x.shape[:2]:
        raise ValueError("phase_mask must have [batch, time] shape")
    valid = phase_mask.to(device=x.device) > 0
    codes = torch.where(valid, codes, torch.zeros_like(codes))
    return embedding(codes) * valid.unsqueeze(-1).to(x.dtype)


class CPETWorldModel(nn.Module):
    """Research MVP for a physiology-guided CPET world model.

    The module keeps the first implementation intentionally compact:
    a masked time-series encoder, a controlled latent dynamics core, endpoint heads,
    and probabilistic trajectory reconstruction.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        self.missingness_projection = nn.Linear(config.n_features, config.d_model, bias=False)
        self.metadata_projection = nn.Linear(config.n_metadata, config.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, config.max_len, config.d_model) * 0.02)
        self.phase_embedding = nn.Embedding(4, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.latent_norm = nn.LayerNorm(config.d_model)
        self.dynamics = nn.GRUCell(config.d_model + 1, config.d_model)
        self.traj_mu = nn.Linear(config.d_model, config.n_features)
        self.traj_logvar = nn.Linear(config.d_model, config.n_features)
        self.at_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1), nn.Sigmoid())
        self.vo2peak_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.stop_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 3))

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = x.shape
        encoded = _mask_aware_encoded_input(x, feature_mask, self.input_projection, self.missingness_projection)
        encoded = encoded + self.position_embedding[:, :steps, :]
        encoded = encoded + _phase_context(x, phase, phase_mask, self.phase_embedding)
        encoded = encoded + self.metadata_projection(metadata).unsqueeze(1)
        key_padding_mask = mask <= 0 if mask is not None else None
        hidden = self.encoder(encoded, src_key_padding_mask=key_padding_mask)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).clamp_min(0.0)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.latent_norm(pooled)

        workload = x[..., -2:-1] if x.shape[-1] >= 2 else torch.zeros(batch, steps, 1, device=x.device, dtype=x.dtype)
        state = pooled
        states = []
        for t in range(steps):
            state = self.dynamics(torch.cat([hidden[:, t, :], workload[:, t, :]], dim=-1), state)
            states.append(state)
        dynamics_hidden = torch.stack(states, dim=1)
        traj_logvar = self.traj_logvar(dynamics_hidden).clamp(-7.0, 5.0)
        return {
            "latent": pooled,
            "trajectory_mu": self.traj_mu(dynamics_hidden),
            "trajectory_logvar": traj_logvar,
            "at_norm": self.at_head(pooled).squeeze(-1),
            "vo2peak": self.vo2peak_head(pooled).squeeze(-1),
            "stop_logits": self.stop_head(pooled),
        }


class DirectTransformerWorldModel(nn.Module):
    """Per-time Transformer teacher with the world-model distillation contract."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.protocol_feature_indices = tuple(config.protocol_feature_indices)
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        self.missingness_projection = nn.Linear(config.n_features, config.d_model, bias=False)
        self.metadata_projection = nn.Linear(config.n_metadata, config.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, config.max_len, config.d_model) * 0.02)
        self.phase_embedding = nn.Embedding(4, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.latent_norm = nn.LayerNorm(config.d_model)
        self.traj_mu = nn.Linear(config.d_model, config.n_features)
        self.traj_logvar = nn.Linear(config.d_model, config.n_features)
        self.at_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1), nn.Sigmoid())
        self.vo2peak_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.stop_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 3))

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = x.shape[1]
        encoded = (
            _mask_aware_encoded_input(x, feature_mask, self.input_projection, self.missingness_projection)
            + self.position_embedding[:, :steps, :]
            + _phase_context(x, phase, phase_mask, self.phase_embedding)
            + self.metadata_projection(metadata).unsqueeze(1)
        )
        hidden = self.encoder(encoded, src_key_padding_mask=(mask <= 0 if mask is not None else None))
        rollout = self.latent_norm(hidden)
        if mask is None:
            pooled = rollout.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).clamp_min(0.0)
            pooled = (rollout * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "latent": pooled,
            "latent_rollout": rollout,
            "trajectory_mu": self.traj_mu(rollout),
            "trajectory_logvar": self.traj_logvar(rollout).clamp(-7.0, 5.0),
            "at_norm": self.at_head(pooled).squeeze(-1),
            "vo2peak": self.vo2peak_head(pooled).squeeze(-1),
            "stop_logits": self.stop_head(pooled),
        }


class ProtocolConditionedLatentStateSpaceWorldModel(nn.Module):
    """Protocol-conditioned latent state-space CPET world model.

    A prefix encoder infers the patient latent state from observed physiology.
    The future trajectory is then rolled out by a recurrent transition model
    conditioned on protocol/action channels such as time, load phase, power, and RPM.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.protocol_feature_indices = tuple(
            index for index in config.protocol_feature_indices if 0 <= index < config.n_features
        )
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        self.missingness_projection = nn.Linear(config.n_features, config.d_model, bias=False)
        self.metadata_projection = nn.Linear(config.n_metadata, config.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, config.max_len, config.d_model) * 0.02)
        self.phase_embedding = nn.Embedding(4, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.latent_norm = nn.LayerNorm(config.d_model)
        protocol_width = len(self.protocol_feature_indices) + (2 * config.d_model)
        self.protocol_projection = nn.Sequential(
            nn.Linear(protocol_width, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.transition = nn.GRUCell(config.d_model, config.d_model)
        self.rollout_norm = nn.LayerNorm(config.d_model)
        self.traj_mu = nn.Linear(config.d_model, config.n_features)
        self.traj_logvar = nn.Linear(config.d_model, config.n_features)
        self.at_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1), nn.Sigmoid())
        self.vo2peak_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.stop_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 3))

    def _protocol_values(self, x: torch.Tensor) -> torch.Tensor:
        if not self.protocol_feature_indices:
            return x.new_zeros((*x.shape[:2], 0))
        return x[..., list(self.protocol_feature_indices)]

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = x.shape
        phase_context = _phase_context(x, phase, phase_mask, self.phase_embedding)
        position_context = self.position_embedding[:, :steps, :].expand(batch, -1, -1)
        encoded = _mask_aware_encoded_input(x, feature_mask, self.input_projection, self.missingness_projection)
        encoded = encoded + position_context
        encoded = encoded + phase_context
        encoded = encoded + self.metadata_projection(metadata).unsqueeze(1)
        key_padding_mask = mask <= 0 if mask is not None else None
        hidden = self.encoder(encoded, src_key_padding_mask=key_padding_mask)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).clamp_min(0.0)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        initial_state = self.latent_norm(pooled)

        protocol_context = self.protocol_projection(
            torch.cat([self._protocol_values(x), phase_context, position_context], dim=-1)
        )
        state = initial_state
        states = []
        for t in range(steps):
            state = self.transition(protocol_context[:, t, :], state)
            states.append(self.rollout_norm(state))
        latent_rollout = torch.stack(states, dim=1)
        endpoint_state = latent_rollout.mean(dim=1)
        traj_logvar = self.traj_logvar(latent_rollout).clamp(-7.0, 5.0)
        return {
            "latent": initial_state,
            "latent_rollout": latent_rollout,
            "trajectory_mu": self.traj_mu(latent_rollout),
            "trajectory_logvar": traj_logvar,
            "at_norm": self.at_head(endpoint_state).squeeze(-1),
            "vo2peak": self.vo2peak_head(endpoint_state).squeeze(-1),
            "stop_logits": self.stop_head(endpoint_state),
        }


class ProtocolConditionedLatentSDEWorldModel(nn.Module):
    """Protocol-conditioned continuous-time latent SDE CPET world model.

    The encoder infers an initial patient state from the observed prefix. The
    rollout core then evolves that state with learned drift and diffusion terms
    conditioned on future protocol/action channels.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.protocol_feature_indices = tuple(
            index for index in config.protocol_feature_indices if 0 <= index < config.n_features
        )
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        self.missingness_projection = nn.Linear(config.n_features, config.d_model, bias=False)
        self.metadata_projection = nn.Linear(config.n_metadata, config.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, config.max_len, config.d_model) * 0.02)
        self.phase_embedding = nn.Embedding(4, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.latent_norm = nn.LayerNorm(config.d_model)
        protocol_width = len(self.protocol_feature_indices) + (2 * config.d_model)
        self.protocol_projection = nn.Sequential(
            nn.Linear(protocol_width, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        dynamics_width = config.d_model * 2
        self.drift = nn.Sequential(
            nn.Linear(dynamics_width, config.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(dynamics_width, config.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.rollout_norm = nn.LayerNorm(config.d_model)
        self.process_to_feature_var = nn.Linear(config.d_model, config.n_features)
        self.traj_mu = nn.Linear(config.d_model, config.n_features)
        self.traj_logvar = nn.Linear(config.d_model, config.n_features)
        self.at_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1), nn.Sigmoid())
        self.vo2peak_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.stop_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 3))

    def _protocol_values(self, x: torch.Tensor) -> torch.Tensor:
        if not self.protocol_feature_indices:
            return x.new_zeros((*x.shape[:2], 0))
        return x[..., list(self.protocol_feature_indices)]

    def _delta_t(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        base = x.new_full((batch, steps, 1), self.config.sde_dt)
        if not self.protocol_feature_indices or steps <= 1:
            return base
        time_values = x[..., self.protocol_feature_indices[0]]
        raw_delta = torch.diff(time_values, dim=1, prepend=time_values[:, :1]).abs().unsqueeze(-1)
        scale = raw_delta[:, 1:].mean(dim=1, keepdim=True).clamp_min(1e-4)
        normalized = raw_delta / scale * self.config.sde_dt
        return normalized.clamp(self.config.sde_dt * 0.25, self.config.sde_dt * 4.0)

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
        phase_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = x.shape
        phase_context = _phase_context(x, phase, phase_mask, self.phase_embedding)
        position_context = self.position_embedding[:, :steps, :].expand(batch, -1, -1)
        metadata_context = self.metadata_projection(metadata).unsqueeze(1)
        encoded = _mask_aware_encoded_input(x, feature_mask, self.input_projection, self.missingness_projection)
        encoded = encoded + position_context
        encoded = encoded + phase_context
        encoded = encoded + metadata_context
        key_padding_mask = mask <= 0 if mask is not None else None
        hidden = self.encoder(encoded, src_key_padding_mask=key_padding_mask)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).clamp_min(0.0)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        state = self.latent_norm(pooled)

        protocol_context = self.protocol_projection(
            torch.cat([self._protocol_values(x), phase_context, position_context], dim=-1)
        )
        delta_t = self._delta_t(x)
        process_var = torch.zeros_like(state)
        states = []
        process_sigmas = []
        process_logvars = []
        for t in range(steps):
            dynamics_input = torch.cat([state, protocol_context[:, t, :]], dim=-1)
            drift = self.drift(dynamics_input)
            sigma = F.softplus(self.diffusion(dynamics_input)).clamp_min(1e-4)
            dt = delta_t[:, t, :]
            if self.training and self.config.sde_noise_scale > 0:
                noise = torch.randn_like(state) * sigma * dt.sqrt() * self.config.sde_noise_scale
            else:
                noise = torch.zeros_like(state)
            state = state + (drift * dt) + noise
            process_var = process_var + sigma.pow(2) * dt
            states.append(self.rollout_norm(state))
            process_sigmas.append(sigma)
            process_logvars.append(process_var.clamp_min(1e-6).log())

        latent_rollout = torch.stack(states, dim=1)
        process_sigma = torch.stack(process_sigmas, dim=1)
        process_logvar = torch.stack(process_logvars, dim=1)
        endpoint_state = latent_rollout.mean(dim=1)
        base_var = self.traj_logvar(latent_rollout).clamp(-7.0, 5.0).exp()
        process_feature_var = F.softplus(self.process_to_feature_var(process_sigma)).clamp_min(1e-5)
        traj_logvar = (base_var + process_feature_var).log().clamp(-7.0, 5.0)
        return {
            "latent": self.latent_norm(pooled),
            "latent_rollout": latent_rollout,
            "process_sigma": process_sigma,
            "process_logvar": process_logvar,
            "trajectory_mu": self.traj_mu(latent_rollout),
            "trajectory_logvar": traj_logvar,
            "at_norm": self.at_head(endpoint_state).squeeze(-1),
            "vo2peak": self.vo2peak_head(endpoint_state).squeeze(-1),
            "stop_logits": self.stop_head(endpoint_state),
        }


PROTOCOL_CONDITIONED_ARCHITECTURES = {
    "protocol_lss",
    "protocol_conditioned_lss",
    "protocol_state_space",
    "protocol_latent_sde",
    "protocol_sde",
    "protocol_conditioned_sde",
}


def _protocol_indices_from_config(config: dict[str, object]) -> tuple[int, ...]:
    raw_indices = config.get("protocol_feature_indices")
    if raw_indices is not None:
        return tuple(int(index) for index in raw_indices)  # type: ignore[iteration-over-annotation]
    feature_columns = list(config.get("feature_columns", ()))  # type: ignore[arg-type]
    protocol_columns: Sequence[str] = config.get(  # type: ignore[assignment]
        "protocol_feature_columns",
        ("Time", "Load_Phase", "Power_Load", "RPM"),
    )
    return tuple(index for index, name in enumerate(feature_columns) if name in set(protocol_columns))


def _model_config(config: dict[str, object], n_features: int, n_metadata: int, max_len: int) -> ModelConfig:
    return ModelConfig(
        n_features=n_features,
        n_metadata=n_metadata,
        max_len=max_len,
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        protocol_feature_indices=_protocol_indices_from_config(config),
        sde_dt=float(config.get("sde_dt", 0.05)),
        sde_noise_scale=float(config.get("sde_noise_scale", 0.0)),
    )


def build_model(config: dict[str, object], n_features: int, n_metadata: int, max_len: int) -> nn.Module:
    model_config = _model_config(config, n_features, n_metadata, max_len)
    architecture = str(config.get("architecture", "masked_completion")).lower()
    if architecture == "protocol_direct_transformer":
        return DirectTransformerWorldModel(model_config)
    if architecture in {"protocol_latent_sde", "protocol_sde", "protocol_conditioned_sde"}:
        return ProtocolConditionedLatentSDEWorldModel(model_config)
    if architecture in {"protocol_lss", "protocol_conditioned_lss", "protocol_state_space"}:
        return ProtocolConditionedLatentStateSpaceWorldModel(model_config)
    return CPETWorldModel(model_config)
