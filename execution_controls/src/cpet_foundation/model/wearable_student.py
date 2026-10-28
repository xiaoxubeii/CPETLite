from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


DERIVED_CPET_FEATURES = (
    "VO2_kg",
    "VCO2_kg",
    "RER",
    "VE_VO2",
    "VE_VCO2",
    "METS",
    "VO2_HR",
)

TEACHER_ONLY_FORBIDDEN_BATCH_KEYS = frozenset(
    {"target_trajectory", "target_channel_mask", "core_derived_confidence", "at_norm", "vo2peak", "vo2peak_mask"}
)
TEACHER_ONLY_REQUIRED_BATCH_KEYS = frozenset(
    {
        "teacher_latent_rollout",
        "teacher_trajectory_mu",
        "teacher_trajectory_logvar",
        "teacher_at_norm",
        "teacher_vo2peak",
    }
)
MEASURED_TARGET_LOSS_KEYS = frozenset(
    {
        "derived_trajectory",
        "direct_trajectory",
        "future_window",
        "peak",
        "slope",
        "boundary_trajectory",
        "morphology",
        "peak_timing",
        "core_derived_trajectory",
        "core_derived_morphology",
        "core_derived_slope",
        "core_derived_peak",
        "core_derived_peak_timing",
        "gas_exchange_log_ratio",
        "rer_slope",
        "gas_exchange_transition",
    }
)


class TemporalConvBlock(nn.Module):
    """Residual dilated temporal block with length-preserving convolutions."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        hidden = values.transpose(1, 2)
        hidden = self.dropout(F.gelu(self.conv1(hidden)))
        hidden = self.dropout(self.conv2(hidden)).transpose(1, 2)
        return self.norm(residual + hidden)


class TemporalConvNet(nn.Module):
    def __init__(self, width: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TemporalConvBlock(width, dilation=2**index, dropout=dropout) for index in range(layers)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return values


def derive_cpet_channels(
    physical: torch.Tensor,
    feature_names: tuple[str, ...],
    body_mass_kg: torch.Tensor | None,
    *,
    min_vo2: float = 50.0,
    min_vco2: float = 50.0,
    min_hr: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite algebraic CPET channels and return their validity mask."""
    required = {"VO2", "VCO2", "VE", "HR", *DERIVED_CPET_FEATURES}
    missing = sorted(required.difference(feature_names))
    if missing:
        raise ValueError(f"deterministic CPET derivation requires features: {missing}")
    index = {name: feature_names.index(name) for name in required}
    result = physical.clone()
    channel_valid = torch.ones_like(result, dtype=torch.bool)
    value_shape = result.shape[:-1]
    if body_mass_kg is None:
        mass = torch.full(value_shape, 70.0, device=result.device, dtype=result.dtype)
        mass_valid = torch.zeros(value_shape, device=result.device, dtype=torch.bool)
    else:
        mass = body_mass_kg.to(device=result.device, dtype=result.dtype)
        while mass.ndim < len(value_shape):
            mass = mass.unsqueeze(-1)
        mass = mass.expand(value_shape)
        mass_valid = torch.isfinite(mass) & (mass >= 20.0) & (mass <= 300.0)
    safe_mass = torch.where(mass_valid, mass, torch.full_like(mass, 70.0))
    vo2 = result[..., index["VO2"]]
    vco2 = result[..., index["VCO2"]]
    ve = result[..., index["VE"]]
    hr = result[..., index["HR"]]
    vo2_valid = torch.isfinite(vo2) & (vo2 >= float(min_vo2))
    vco2_valid = torch.isfinite(vco2) & (vco2 >= float(min_vco2))
    hr_valid = torch.isfinite(hr) & (hr >= float(min_hr))
    safe_vo2 = vo2.clamp_min(float(min_vo2))
    safe_vco2 = vco2.clamp_min(float(min_vco2))
    safe_hr = hr.clamp_min(float(min_hr))
    assignments = {
        "VO2_kg": (vo2 / safe_mass, mass_valid),
        "VCO2_kg": (vco2 / safe_mass, mass_valid),
        "RER": (vco2 / safe_vo2, vo2_valid & vco2_valid),
        "VE_VO2": (1000.0 * ve / safe_vo2, vo2_valid),
        "VE_VCO2": (1000.0 * ve / safe_vco2, vco2_valid),
        "METS": (vo2 / safe_mass / 3.5, mass_valid),
        "VO2_HR": (vo2 / safe_hr, hr_valid),
    }
    result_channels = []
    valid_channels = []
    for channel_index, name in enumerate(feature_names):
        if name in assignments:
            values, valid = assignments[name]
            result_channels.append(values)
            valid_channels.append(valid)
        else:
            result_channels.append(result[..., channel_index])
            valid_channels.append(channel_valid[..., channel_index])
    return torch.stack(result_channels, dim=-1), torch.stack(valid_channels, dim=-1)


@dataclass(frozen=True)
class WearableStudentConfig:
    n_wearable_features: int
    n_metadata: int
    n_target_features: int
    max_len: int
    n_protocol_features: int = 0
    temporal_backbone_type: str = "transformer"
    rollout_type: str = "neural_sde"
    sde_dt: float = 1.0
    sde_noise_scale: float = 0.08
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    deterministic_derived_channels: bool = False
    target_feature_names: tuple[str, ...] = ()
    base_feature_names: tuple[str, ...] = ()
    output_feature_mean: tuple[float, ...] = ()
    output_feature_std: tuple[float, ...] = ()
    metadata_feature_mean: tuple[float, ...] = ()
    metadata_feature_std: tuple[float, ...] = ()
    initial_anchor_steps: int = 8
    derived_min_vo2: float = 50.0
    derived_min_vco2: float = 50.0
    derived_min_hr: float = 30.0
    joint_gas_exchange_decoder: bool = False
    input_ablation: str = "none"


class WearableCpetStudentModel(nn.Module):
    """Wearable-to-CPET student with CPET world-model compatible outputs."""

    def __init__(self, config: WearableStudentConfig) -> None:
        super().__init__()
        if config.input_ablation not in {"none", "protocol_static"}:
            raise ValueError(f"unsupported input_ablation: {config.input_ablation}")
        supported_rollouts = {"neural_sde", "latent_ode_euler", "latent_ode_heun", "gru", "direct"}
        if config.rollout_type not in supported_rollouts:
            raise ValueError(
                f"unsupported rollout_type: {config.rollout_type}; "
                f"expected one of {sorted(supported_rollouts)}"
            )
        self.config = config
        self.wearable_encoder = nn.Sequential(
            nn.Linear(config.n_wearable_features * 2, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.protocol_encoder = (
            nn.Sequential(
                nn.Linear(config.n_protocol_features * 2, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            if config.n_protocol_features > 0
            else None
        )
        self.protocol_gate = nn.Linear(config.d_model * 2, config.d_model) if config.n_protocol_features > 0 else None
        self.metadata_projection = nn.Linear(config.n_metadata, config.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, config.max_len, config.d_model) * 0.02)
        self.phase_embedding = nn.Embedding(4, config.d_model)
        backbone_type = config.temporal_backbone_type.lower()
        if backbone_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_backbone = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        elif backbone_type == "tcn":
            self.temporal_backbone = TemporalConvNet(config.d_model, config.n_layers, config.dropout)
        elif backbone_type == "gru":
            if config.d_model % 2:
                raise ValueError("bidirectional GRU requires an even d_model")
            self.temporal_backbone = nn.GRU(
                config.d_model, config.d_model // 2, num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0.0,
                batch_first=True, bidirectional=True,
            )
        else:
            raise ValueError(f"unsupported temporal_backbone_type: {config.temporal_backbone_type}")
        self.latent_norm = nn.LayerNorm(config.d_model)
        self.rollout_transition = nn.GRUCell(config.d_model, config.d_model)
        self.sde_drift = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
        )
        self.sde_diffusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.Softplus(),
        )
        self.rollout_norm = nn.LayerNorm(config.d_model)
        self.target_feature_names = tuple(config.target_feature_names)
        self.base_feature_names = tuple(config.base_feature_names)
        if config.deterministic_derived_channels:
            if len(self.target_feature_names) != config.n_target_features:
                raise ValueError("deterministic derived channels require one target_feature_name per output")
            missing = [name for name in self.base_feature_names if name not in self.target_feature_names]
            if missing:
                raise ValueError(f"base features are absent from target features: {missing}")
            decoder_width = len(self.base_feature_names)
            if decoder_width == 0:
                raise ValueError("deterministic derived channels require base_feature_names")
        else:
            decoder_width = config.n_target_features
        self.base_feature_indices = tuple(
            self.target_feature_names.index(name) for name in self.base_feature_names
        ) if config.deterministic_derived_channels else tuple(range(config.n_target_features))
        self.traj_mu = nn.Linear(config.d_model, decoder_width)
        self.traj_logvar = nn.Linear(config.d_model, decoder_width)
        self.gas_exchange_base_indices: tuple[int, int] = ()
        self.gas_exchange_mu: nn.Module | None = None
        self.gas_exchange_logvar: nn.Module | None = None
        if config.joint_gas_exchange_decoder:
            if not config.deterministic_derived_channels:
                raise ValueError("joint gas-exchange decoding requires deterministic base channels")
            missing_gas = [name for name in ("VO2", "VCO2") if name not in self.base_feature_names]
            if missing_gas:
                raise ValueError(f"joint gas-exchange decoding requires base features: {missing_gas}")
            self.gas_exchange_base_indices = (
                self.base_feature_names.index("VO2"),
                self.base_feature_names.index("VCO2"),
            )
            self.gas_exchange_mu = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
            self.gas_exchange_logvar = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
            )
        self.initial_mu = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, decoder_width),
        )
        initial_mean = torch.zeros(config.n_target_features, dtype=torch.float32)
        initial_std = torch.ones(config.n_target_features, dtype=torch.float32)
        if len(config.output_feature_mean) == config.n_target_features:
            initial_mean = torch.tensor(config.output_feature_mean, dtype=torch.float32)
        if len(config.output_feature_std) == config.n_target_features:
            initial_std = torch.tensor(config.output_feature_std, dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("output_feature_mean", initial_mean)
        self.register_buffer("output_feature_std", initial_std)
        metadata_mean = torch.zeros(config.n_metadata, dtype=torch.float32)
        metadata_std = torch.ones(config.n_metadata, dtype=torch.float32)
        if len(config.metadata_feature_mean) == config.n_metadata:
            metadata_mean = torch.tensor(config.metadata_feature_mean, dtype=torch.float32)
        if len(config.metadata_feature_std) == config.n_metadata:
            metadata_std = torch.tensor(config.metadata_feature_std, dtype=torch.float32).clamp_min(1e-6)
        self.register_buffer("metadata_feature_mean", metadata_mean)
        self.register_buffer("metadata_feature_std", metadata_std)
        self.at_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1), nn.Sigmoid())
        self.vo2peak_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.stop_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 3))

    def forward(
        self,
        wearable_x: torch.Tensor,
        metadata: torch.Tensor,
        wearable_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        protocol_x: torch.Tensor | None = None,
        protocol_mask: torch.Tensor | None = None,
        body_mass_kg: torch.Tensor | None = None,
        sample_count: int = 0,
        phase_mask: torch.Tensor | None = None,
        metadata_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = wearable_x.shape
        if steps > self.config.max_len:
            raise ValueError(f"wearable sequence length {steps} exceeds max_len {self.config.max_len}")

        feature_mask = torch.ones_like(wearable_x) if wearable_mask is None else wearable_mask.to(wearable_x.dtype)
        if self.config.input_ablation == "protocol_static":
            # Remove both values and missingness/quality side information. Keep
            # protocol, static covariates, phase and the original time axis.
            wearable_x = torch.zeros_like(wearable_x)
            feature_mask = torch.zeros_like(feature_mask)
        observed = torch.where(feature_mask > 0, wearable_x, torch.zeros_like(wearable_x)) * feature_mask
        encoded = self.wearable_encoder(torch.cat([observed, feature_mask], dim=-1))
        protocol_feature_mask = None
        if protocol_x is not None and self.protocol_encoder is not None:
            protocol_feature_mask = torch.ones_like(protocol_x) if protocol_mask is None else protocol_mask.to(protocol_x.dtype)
            protocol_observed = torch.where(
                protocol_feature_mask > 0, protocol_x, torch.zeros_like(protocol_x)
            ) * protocol_feature_mask
            protocol_encoded = self.protocol_encoder(torch.cat([protocol_observed, protocol_feature_mask], dim=-1))
            gate = torch.sigmoid(self.protocol_gate(torch.cat([encoded, protocol_encoded], dim=-1)))
            encoded = encoded + gate * protocol_encoded
        phase_raw = phase.clamp(0, 3).long() if phase is not None else torch.zeros_like(wearable_x[..., 0], dtype=torch.long)
        encoded = encoded + self.position_embedding[:, :steps, :]
        phase_context = self.phase_embedding(phase_raw)
        if phase_mask is not None:
            if phase_mask.shape != phase_raw.shape:
                raise ValueError("phase_mask must match the time axis")
            phase_context = phase_context * (phase_mask > 0).unsqueeze(-1).to(encoded.dtype)
        encoded = encoded + phase_context
        encoded = encoded + self.metadata_projection(self._normalize_metadata(metadata, metadata_mask)).unsqueeze(1)

        sequence_mask = self._sequence_mask(feature_mask, mask, protocol_feature_mask)
        attention_mask = self._attention_mask(sequence_mask)
        if self.config.temporal_backbone_type.lower() == "transformer":
            hidden = self.temporal_backbone(encoded, src_key_padding_mask=attention_mask <= 0)
        elif self.config.temporal_backbone_type.lower() == "gru":
            # Pack the valid time axis, not the count of observed signal values:
            # a missing internal observation still occupies its original time.
            valid_time = mask if mask is not None else torch.ones_like(sequence_mask)
            positions = torch.arange(1, steps + 1, device=encoded.device)
            lengths = (valid_time.gt(0) * positions).amax(dim=1).clamp_min(1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded * sequence_mask.unsqueeze(-1).to(encoded.dtype), lengths,
                batch_first=True, enforce_sorted=False,
            )
            packed_hidden, _ = self.temporal_backbone(packed)
            hidden, _ = nn.utils.rnn.pad_packed_sequence(
                packed_hidden, batch_first=True, total_length=steps,
            )
        else:
            hidden = self.temporal_backbone(encoded * sequence_mask.unsqueeze(-1).to(encoded.dtype))
        weights = sequence_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        latent = self.latent_norm(pooled)

        if self.config.rollout_type == "neural_sde":
            latent_rollout, diffusion_rollout = self._mean_sde_rollout(latent, hidden)
        elif self.config.rollout_type in {"latent_ode_euler", "latent_ode_heun"}:
            method = self.config.rollout_type.removeprefix("latent_ode_")
            latent_rollout = self._latent_ode_rollout(latent, hidden, method=method)
            diffusion_rollout = torch.zeros_like(latent_rollout)
        elif self.config.rollout_type == "direct":
            latent_rollout = self.rollout_norm(hidden)
            diffusion_rollout = torch.zeros_like(latent_rollout)
        elif self.config.rollout_type == "gru":
            latent_rollout = self._gru_rollout(latent, hidden)
            diffusion_rollout = torch.zeros_like(latent_rollout)
        else:  # guarded during construction
            raise AssertionError(f"unreachable rollout_type: {self.config.rollout_type}")
        endpoint_state = (latent_rollout * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        base_mu, base_logvar = self._decode_base(latent_rollout)
        initial_mu = self.initial_mu(torch.cat([latent, hidden[:, 0, :]], dim=-1))
        base_mu = self._anchor_initial_steps(base_mu, initial_mu, sequence_mask)
        trajectory_mu, trajectory_logvar = self._assemble_trajectory(base_mu, base_logvar, body_mass_kg)
        outputs = {
            "rollout_type": self.config.rollout_type,
            "latent": latent,
            "latent_rollout": latent_rollout,
            "sde_diffusion": diffusion_rollout,
            "trajectory_mu": trajectory_mu,
            "trajectory_logvar": trajectory_logvar,
            "at_norm": self.at_head(endpoint_state).squeeze(-1),
            "vo2peak": self.vo2peak_head(endpoint_state).squeeze(-1),
            "stop_logits": self.stop_head(endpoint_state),
        }
        if sample_count > 0 and self.config.rollout_type == "neural_sde":
            latent_samples = self._sample_sde_rollout(latent, hidden, sample_count)
            outputs["latent_sample_rollout"] = latent_samples
            sample_base_mu, sample_base_logvar = self._decode_base(latent_samples)
            sample_base_mu = self._anchor_initial_steps(sample_base_mu, initial_mu.unsqueeze(1), sequence_mask)
            sample_mass = body_mass_kg
            if sample_mass is not None:
                sample_mass = sample_mass.unsqueeze(1).expand(-1, sample_count)
            outputs["trajectory_samples"], _ = self._assemble_trajectory(
                sample_base_mu, sample_base_logvar, sample_mass
            )
        return outputs

    def _decode_base(self, latent_rollout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_mu = self.traj_mu(latent_rollout)
        base_logvar = self.traj_logvar(latent_rollout)
        if self.gas_exchange_mu is not None and self.gas_exchange_logvar is not None:
            gas_mu = self.gas_exchange_mu(latent_rollout)
            gas_logvar = self.gas_exchange_logvar(latent_rollout)
            gas_by_index = {
                self.gas_exchange_base_indices[0]: 0,
                self.gas_exchange_base_indices[1]: 1,
            }
            base_mu = torch.stack(
                [gas_mu[..., gas_by_index[index]] if index in gas_by_index else base_mu[..., index]
                 for index in range(base_mu.shape[-1])],
                dim=-1,
            )
            base_logvar = torch.stack(
                [gas_logvar[..., gas_by_index[index]] if index in gas_by_index else base_logvar[..., index]
                 for index in range(base_logvar.shape[-1])],
                dim=-1,
            )
        return base_mu, base_logvar.clamp(-7.0, 5.0)

    def _normalize_metadata(self, metadata: torch.Tensor, metadata_mask: torch.Tensor | None = None) -> torch.Tensor:
        mean = self.metadata_feature_mean.to(metadata)
        std = self.metadata_feature_std.to(metadata).clamp_min(1e-6)
        valid = torch.isfinite(metadata)
        if metadata_mask is not None:
            if metadata_mask.shape != metadata.shape:
                raise ValueError("metadata_mask must match the metadata fields")
            valid = valid & (metadata_mask > 0)
        elif metadata.shape[-1] == 3:
            # Historical three-field callers used positive-only demographics.
            # Current readers always pass field-level validity, including Gender=0.
            valid = valid & (metadata > 0)
        imputed = torch.where(valid, metadata, mean)
        return (imputed - mean) / std

    def _anchor_initial_steps(
        self, base_mu: torch.Tensor, initial_mu: torch.Tensor, sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(base_mu.shape[-2], device=base_mu.device, dtype=base_mu.dtype)
        lengths = ((positions + 1) * (sequence_mask > 0)).amax(dim=1).clamp_min(1)
        steps = lengths.clamp_max(max(int(self.config.initial_anchor_steps), 1))
        decay = (1.0 - positions.unsqueeze(0) / (steps - 1).clamp_min(1).unsqueeze(1)).clamp_min(0.0)
        while initial_mu.ndim < base_mu.ndim - 1:
            initial_mu = initial_mu.unsqueeze(1)
        offset = initial_mu - base_mu[..., 0, :]
        shape = [base_mu.shape[0]] + [1] * (base_mu.ndim - 3) + [base_mu.shape[-2], 1]
        return base_mu + offset.unsqueeze(-2) * decay.reshape(shape)

    def set_output_scaler(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.config.n_target_features or std.numel() != self.config.n_target_features:
            raise ValueError("output scaler width must equal n_target_features")
        self.output_feature_mean.copy_(mean.detach().reshape(-1).to(self.output_feature_mean))
        self.output_feature_std.copy_(std.detach().reshape(-1).to(self.output_feature_std).clamp_min(1e-6))

    def _assemble_trajectory(
        self,
        base_mu: torch.Tensor,
        base_logvar: torch.Tensor,
        body_mass_kg: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.deterministic_derived_channels:
            return base_mu, base_logvar
        mean = self.output_feature_mean.to(base_mu)
        std = self.output_feature_std.to(base_mu).clamp_min(1e-6)
        base_indices = torch.tensor(self.base_feature_indices, device=base_mu.device, dtype=torch.long)
        base_physical = (base_mu * std.index_select(0, base_indices) + mean.index_select(0, base_indices)).clamp_min(0.0)
        physical_shape = (*base_mu.shape[:-1], self.config.n_target_features)
        physical = torch.zeros(physical_shape, device=base_mu.device, dtype=base_mu.dtype)
        physical.index_copy_(-1, base_indices, base_physical)
        physical, _ = derive_cpet_channels(
            physical,
            self.target_feature_names,
            body_mass_kg,
            min_vo2=self.config.derived_min_vo2,
            min_vco2=self.config.derived_min_vco2,
            min_hr=self.config.derived_min_hr,
        )
        full_mu = (physical - mean) / std
        full_logvar = torch.zeros_like(full_mu)
        full_logvar.index_copy_(-1, base_indices, base_logvar)
        return full_mu, full_logvar

    def _gru_rollout(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        state = latent
        states = []
        for step in range(hidden.shape[1]):
            state = self.rollout_transition(hidden[:, step, :], state)
            states.append(self.rollout_norm(state))
        return torch.stack(states, dim=1)

    def _latent_ode_rollout(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        *,
        method: str,
    ) -> torch.Tensor:
        if method not in {"euler", "heun"}:
            raise ValueError(f"unsupported latent ODE method: {method}")
        state = latent
        states = []
        dt = float(self.config.sde_dt)
        for step in range(hidden.shape[1]):
            control = hidden[:, step, :]
            drift = self.sde_drift(torch.cat([state, control], dim=-1))
            if method == "euler":
                state = state + drift * dt
            else:
                predictor = state + drift * dt
                corrected_drift = self.sde_drift(torch.cat([predictor, control], dim=-1))
                state = state + 0.5 * (drift + corrected_drift) * dt
            states.append(self.rollout_norm(state))
        return torch.stack(states, dim=1)

    def _mean_sde_rollout(self, latent: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = latent
        states = []
        diffusions = []
        dt = float(self.config.sde_dt)
        for step in range(hidden.shape[1]):
            context = torch.cat([state, hidden[:, step, :]], dim=-1)
            drift = self.sde_drift(context)
            diffusion = self.sde_diffusion(context) * float(self.config.sde_noise_scale)
            state = state + drift * dt
            states.append(self.rollout_norm(state))
            diffusions.append(diffusion)
        return torch.stack(states, dim=1), torch.stack(diffusions, dim=1)

    def _sample_sde_rollout(self, latent: torch.Tensor, hidden: torch.Tensor, sample_count: int) -> torch.Tensor:
        batch, steps, width = hidden.shape
        state = latent.unsqueeze(1).expand(batch, sample_count, width).contiguous()
        controls = hidden.unsqueeze(1).expand(batch, sample_count, steps, width)
        states = []
        dt = float(self.config.sde_dt)
        sqrt_dt = max(dt, 1e-6) ** 0.5
        for step in range(steps):
            context = torch.cat([state, controls[:, :, step, :]], dim=-1)
            drift = self.sde_drift(context)
            diffusion = self.sde_diffusion(context) * float(self.config.sde_noise_scale)
            noise = torch.randn_like(state)
            state = state + drift * dt + diffusion * sqrt_dt * noise
            states.append(self.rollout_norm(state))
        return torch.stack(states, dim=2)

    @staticmethod
    def _sequence_mask(
        wearable_mask: torch.Tensor,
        mask: torch.Tensor | None,
        protocol_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_observed = (wearable_mask.sum(dim=-1) > 0).to(wearable_mask.dtype)
        if protocol_mask is not None:
            protocol_observed = (protocol_mask.sum(dim=-1) > 0).to(wearable_mask.dtype)
            feature_observed = torch.maximum(feature_observed, protocol_observed)
        if mask is None:
            return feature_observed
        return feature_observed * mask.to(wearable_mask.dtype)

    @staticmethod
    def _attention_mask(sequence_mask: torch.Tensor) -> torch.Tensor:
        empty_rows = sequence_mask.sum(dim=1) <= 0
        first_step = torch.arange(sequence_mask.shape[1], device=sequence_mask.device).unsqueeze(0) == 0
        fallback = first_step.expand_as(sequence_mask).to(sequence_mask.dtype)
        return torch.where(empty_rows.unsqueeze(1), fallback, sequence_mask)


def wearable_student_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
    *,
    supervision_mode: str = "hybrid",
    trajectory_distillation_mode: str = "gaussian_nll",
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_weights = {
        "latent_alignment": 1.0,
        "trajectory_distillation": 1.0,
        "endpoint": 1.0,
        "teacher_endpoint": 0.0,
        "teacher_at_endpoint": 0.0,
        "at_region_distillation": 0.0,
        "relation_distillation": 0.0,
        "teacher_morphology_distillation": 0.0,
        "selective_distillation": 0.0,
        "uncertainty_distillation": 0.25,
        "physiology_consistency": 0.1,
        "derived_consistency": 0.0,
        "derived_trajectory": 0.0,
        "direct_trajectory": 0.0,
        "future_window": 0.0,
        "peak": 0.0,
        "slope": 0.0,
        "boundary_trajectory": 0.0,
        "morphology": 0.0,
        "peak_timing": 0.0,
        "core_derived_trajectory": 0.0,
        "core_derived_morphology": 0.0,
        "core_derived_slope": 0.0,
        "core_derived_peak": 0.0,
        "core_derived_peak_timing": 0.0,
        "gas_exchange_log_ratio": 0.0,
        "rer_slope": 0.0,
        "gas_exchange_transition": 0.0,
    }
    if weights is not None:
        loss_weights = {**loss_weights, **dict(weights)}

    normalized_supervision_mode = supervision_mode.strip().lower()
    if normalized_supervision_mode not in {"hybrid", "teacher_only"}:
        raise ValueError(f"unsupported supervision_mode: {supervision_mode!r}")
    if normalized_supervision_mode == "teacher_only":
        exposed = sorted(TEACHER_ONLY_FORBIDDEN_BATCH_KEYS.intersection(batch))
        if exposed:
            raise ValueError(f"teacher-only batch contains forbidden measured target(s): {exposed}")
        missing = sorted(TEACHER_ONLY_REQUIRED_BATCH_KEYS.difference(batch))
        if missing:
            raise ValueError(f"teacher-only batch is missing teacher target(s): {missing}")
        active_measured_losses = {
            name: float(loss_weights[name])
            for name in MEASURED_TARGET_LOSS_KEYS
            if abs(float(loss_weights.get(name, 0.0))) > 0.0
        }
        if active_measured_losses:
            raise ValueError(
                "teacher-only supervision requires measured-target loss weights to be zero: "
                f"{active_measured_losses}"
            )

    mask = batch.get("mask")
    if mask is None:
        mask = torch.ones(outputs["trajectory_mu"].shape[:2], device=outputs["trajectory_mu"].device)
    mask = mask.to(outputs["trajectory_mu"].dtype)
    if trajectory_distillation_mode not in {"gaussian_nll", "mean_mse"}:
        raise ValueError(f"unsupported trajectory_distillation_mode: {trajectory_distillation_mode}")
    if trajectory_distillation_mode == "mean_mse":
        if normalized_supervision_mode != "teacher_only":
            raise ValueError("mean_mse control requires teacher_only supervision")
        active_auxiliary = {key: value for key, value in loss_weights.items()
                            if key not in {"trajectory_distillation", "endpoint"} and float(value) != 0}
        if active_auxiliary:
            raise ValueError(f"mean_mse control requires auxiliary losses disabled: {active_auxiliary}")
        mu, target = _align_last_dim(outputs["trajectory_mu"], batch["teacher_trajectory_mu"])
        indices = _loss_feature_indices(batch, "distillation_feature_indices", mu.shape[-1], mu.device)
        if indices.numel() > 0:
            mu, target = mu.index_select(-1, indices), target.index_select(-1, indices)
        valid = (mask.unsqueeze(-1) > 0) & torch.isfinite(target)
        if bool((valid & ~torch.isfinite(mu)).any()):
            raise FloatingPointError("Non-finite prediction at an eligible teacher mean target")
        error = (torch.where(valid, mu, torch.zeros_like(mu))
                 - torch.where(valid, target.to(mu.dtype), torch.zeros_like(mu))).square()
        trajectory = error.sum() / valid.sum().clamp_min(1)
        endpoint = _endpoint_supervision(outputs, batch, target_source="teacher")
        total = loss_weights["trajectory_distillation"] * trajectory + loss_weights["endpoint"] * endpoint
        return total, {"loss": float(total.detach().cpu()),
                       "trajectory_distillation": float(trajectory.detach().cpu()),
                       "endpoint": float(endpoint.detach().cpu())}
    measured_mask = mask
    trajectory_supervision_mask = batch.get("trajectory_supervision_mask")
    if trajectory_supervision_mask is not None:
        measured_mask = measured_mask * trajectory_supervision_mask.to(mask.dtype).reshape(-1, 1)

    latent_alignment = _masked_mse_optional(
        outputs.get("latent_rollout"),
        batch.get("teacher_latent_rollout"),
        mask,
    )
    trajectory_distillation = _trajectory_distillation(
        outputs,
        batch,
        mask,
        use_observability_weights=bool(float(loss_weights.get("observability_weighted_distillation", 0.0))),
    )
    endpoint = _endpoint_supervision(outputs, batch, target_source="auto")
    teacher_endpoint = _endpoint_supervision(outputs, batch, target_source="teacher")
    teacher_at_endpoint = _teacher_at_supervision(outputs, batch)
    at_region_distillation = _at_region_distillation(outputs, batch, mask)
    relation_distillation = _relation_distillation(outputs, batch, mask)
    teacher_morphology_distillation = _teacher_morphology_distillation(outputs, batch, mask)
    selective_distillation = _selective_teacher_distillation(outputs, batch, mask)
    uncertainty_target = _teacher_logvar_target(batch)
    uncertainty_distillation = _selected_mse_optional(
        outputs.get("trajectory_logvar"), uncertainty_target, mask, batch, "distillation_feature_indices"
    )
    physiology_consistency = _physiology_consistency(outputs["trajectory_mu"], batch, mask)
    derived_consistency = _derived_consistency(outputs["trajectory_mu"], batch, mask)
    derived_trajectory = _selected_trajectory_loss(
        outputs["trajectory_mu"], batch, measured_mask, "derived_loss_feature_indices"
    )
    direct_trajectory, peak, slope = _measured_curve_losses(outputs["trajectory_mu"], batch, measured_mask)
    boundary_trajectory, morphology, peak_timing = _boundary_morphology_timing_losses(
        outputs["trajectory_mu"], batch, measured_mask
    )
    (
        core_derived_trajectory,
        core_derived_morphology,
        core_derived_slope,
        core_derived_peak,
        core_derived_peak_timing,
    ) = _core_derived_curve_losses(outputs["trajectory_mu"], batch, measured_mask)
    gas_exchange_log_ratio, rer_slope, gas_exchange_transition = _joint_gas_exchange_losses(
        outputs["trajectory_mu"], batch, measured_mask
    )

    total = (
        loss_weights["latent_alignment"] * latent_alignment
        + loss_weights["trajectory_distillation"] * trajectory_distillation
        + loss_weights["endpoint"] * endpoint
        + loss_weights["teacher_endpoint"] * teacher_endpoint
        + loss_weights["teacher_at_endpoint"] * teacher_at_endpoint
        + loss_weights["at_region_distillation"] * at_region_distillation
        + loss_weights["relation_distillation"] * relation_distillation
        + loss_weights["teacher_morphology_distillation"] * teacher_morphology_distillation
        + loss_weights["selective_distillation"] * selective_distillation
        + loss_weights["uncertainty_distillation"] * uncertainty_distillation
        + loss_weights["physiology_consistency"] * physiology_consistency
        + loss_weights["derived_consistency"] * derived_consistency
        + loss_weights["derived_trajectory"] * derived_trajectory
        + loss_weights["direct_trajectory"] * direct_trajectory
        + loss_weights["future_window"] * direct_trajectory
        + loss_weights["peak"] * peak
        + loss_weights["slope"] * slope
        + loss_weights["boundary_trajectory"] * boundary_trajectory
        + loss_weights["morphology"] * morphology
        + loss_weights["peak_timing"] * peak_timing
        + loss_weights["core_derived_trajectory"] * core_derived_trajectory
        + loss_weights["core_derived_morphology"] * core_derived_morphology
        + loss_weights["core_derived_slope"] * core_derived_slope
        + loss_weights["core_derived_peak"] * core_derived_peak
        + loss_weights["core_derived_peak_timing"] * core_derived_peak_timing
        + loss_weights["gas_exchange_log_ratio"] * gas_exchange_log_ratio
        + loss_weights["rer_slope"] * rer_slope
        + loss_weights["gas_exchange_transition"] * gas_exchange_transition
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "latent_alignment": float(latent_alignment.detach().cpu()),
        "trajectory_distillation": float(trajectory_distillation.detach().cpu()),
        "endpoint": float(endpoint.detach().cpu()),
        "teacher_endpoint": float(teacher_endpoint.detach().cpu()),
        "teacher_at_endpoint": float(teacher_at_endpoint.detach().cpu()),
        "at_region_distillation": float(at_region_distillation.detach().cpu()),
        "relation_distillation": float(relation_distillation.detach().cpu()),
        "teacher_morphology_distillation": float(teacher_morphology_distillation.detach().cpu()),
        "selective_distillation": float(selective_distillation.detach().cpu()),
        "uncertainty_distillation": float(uncertainty_distillation.detach().cpu()),
        "physiology_consistency": float(physiology_consistency.detach().cpu()),
        "derived_consistency": float(derived_consistency.detach().cpu()),
        "derived_trajectory": float(derived_trajectory.detach().cpu()),
        "direct_trajectory": float(direct_trajectory.detach().cpu()),
        "future_window": float(direct_trajectory.detach().cpu()),
        "peak": float(peak.detach().cpu()),
        "slope": float(slope.detach().cpu()),
        "boundary_trajectory": float(boundary_trajectory.detach().cpu()),
        "morphology": float(morphology.detach().cpu()),
        "peak_timing": float(peak_timing.detach().cpu()),
        "core_derived_trajectory": float(core_derived_trajectory.detach().cpu()),
        "core_derived_morphology": float(core_derived_morphology.detach().cpu()),
        "core_derived_slope": float(core_derived_slope.detach().cpu()),
        "core_derived_peak": float(core_derived_peak.detach().cpu()),
        "core_derived_peak_timing": float(core_derived_peak_timing.detach().cpu()),
        "gas_exchange_log_ratio": float(gas_exchange_log_ratio.detach().cpu()),
        "rer_slope": float(rer_slope.detach().cpu()),
        "gas_exchange_transition": float(gas_exchange_transition.detach().cpu()),
    }


def _joint_gas_exchange_losses(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise the VO2-VCO2 relationship in physical units."""
    zero = torch.zeros((), device=mu.device, dtype=mu.dtype)
    target = batch.get("target_trajectory")
    mean = batch.get("target_feature_mean")
    std = batch.get("target_feature_std")
    raw_indices = batch.get("gas_exchange_feature_indices")
    if target is None or mean is None or std is None or raw_indices is None:
        return zero, zero, zero
    indices = raw_indices.to(device=mu.device, dtype=torch.long).flatten()
    if indices.numel() != 2 or bool((indices < 0).any()) or bool((indices >= mu.shape[-1]).any()):
        return zero, zero, zero
    mu, target = _align_last_dim(mu, target.to(mu.dtype))
    mean = mean.to(device=mu.device, dtype=mu.dtype)
    std = std.to(device=mu.device, dtype=mu.dtype).clamp_min(1e-6)
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
    if std.ndim == 1:
        std = std.unsqueeze(0)
    if mean.shape[-1] != mu.shape[-1] or std.shape[-1] != mu.shape[-1]:
        return zero, zero, zero
    pred_physical = mu * std.unsqueeze(1) + mean.unsqueeze(1)
    target_physical = target * std.unsqueeze(1) + mean.unsqueeze(1)
    vo2_index, vco2_index = (int(indices[0]), int(indices[1]))
    pred_vo2 = pred_physical[..., vo2_index]
    pred_vco2 = pred_physical[..., vco2_index]
    target_vo2 = target_physical[..., vo2_index]
    target_vco2 = target_physical[..., vco2_index]

    min_vo2 = float(batch.get("gas_exchange_min_vo2", torch.tensor(50.0)).reshape(-1)[0].item())
    min_vco2 = float(batch.get("gas_exchange_min_vco2", torch.tensor(50.0)).reshape(-1)[0].item())
    valid = mask > 0
    channel_mask = batch.get("target_channel_mask")
    if channel_mask is not None:
        valid = valid & (channel_mask[..., vo2_index] > 0) & (channel_mask[..., vco2_index] > 0)
    valid = valid & torch.isfinite(pred_vo2) & torch.isfinite(pred_vco2)
    valid = valid & torch.isfinite(target_vo2) & torch.isfinite(target_vco2)
    valid = valid & (target_vo2 >= min_vo2) & (target_vco2 >= min_vco2)
    valid = valid & (pred_vo2 >= min_vo2) & (pred_vco2 >= min_vco2)
    valid_f = valid.to(mu.dtype)

    pred_log_ratio = torch.log(pred_vco2.clamp_min(min_vco2)) - torch.log(pred_vo2.clamp_min(min_vo2))
    target_log_ratio = torch.log(target_vco2.clamp_min(min_vco2)) - torch.log(target_vo2.clamp_min(min_vo2))
    ratio_error = F.smooth_l1_loss(pred_log_ratio, target_log_ratio, reduction="none")
    ratio_loss = (ratio_error * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    pred_rer = pred_vco2.clamp_min(min_vco2) / pred_vo2.clamp_min(min_vo2)
    target_rer = target_vco2.clamp_min(min_vco2) / target_vo2.clamp_min(min_vo2)
    if mu.shape[1] > 1:
        slope_valid = (valid[:, 1:] & valid[:, :-1]).to(mu.dtype)
        slope_error = F.smooth_l1_loss(
            pred_rer[:, 1:] - pred_rer[:, :-1],
            target_rer[:, 1:] - target_rer[:, :-1],
            reduction="none",
        )
        slope_loss = (slope_error * slope_valid).sum() / slope_valid.sum().clamp_min(1.0)
    else:
        slope_loss = zero

    positions = _physical_time_fraction(batch, mu, mask)
    at_norm = batch.get("at_norm")
    if at_norm is None:
        transition_loss = zero
    else:
        at, at_valid = _valid_measured_at(batch, mu)
        at = at.reshape(-1, 1)
        bandwidth = float(batch.get("gas_exchange_at_bandwidth", torch.tensor(0.1)).reshape(-1)[0].item())
        at_weight = float(batch.get("gas_exchange_at_weight", torch.tensor(1.0)).reshape(-1)[0].item())
        post_weight = float(batch.get("gas_exchange_post_at_weight", torch.tensor(1.0)).reshape(-1)[0].item())
        local = torch.exp(-0.5 * ((positions - at) / max(bandwidth, 1e-3)).square())
        transition_weight = 1.0 + max(at_weight - 1.0, 0.0) * local
        transition_weight = transition_weight + max(post_weight - 1.0, 0.0) * (positions >= at).to(mu.dtype)
        weighted_valid = valid_f * transition_weight * at_valid.reshape(-1, 1)
        transition_loss = (ratio_error * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
    return ratio_loss, slope_loss, transition_loss


def build_wearable_student_from_config(
    config: Mapping[str, object] | WearableStudentConfig | None,
    n_wearable_features: int,
    n_metadata: int,
    n_target_features: int,
    max_len: int,
    n_protocol_features: int = 0,
) -> WearableCpetStudentModel:
    if isinstance(config, WearableStudentConfig):
        return WearableCpetStudentModel(config)

    provided = dict(config or {})
    valid_fields = {field.name for field in fields(WearableStudentConfig)}
    values = {
        "n_wearable_features": n_wearable_features,
        "n_protocol_features": n_protocol_features,
        "n_metadata": n_metadata,
        "n_target_features": n_target_features,
        "max_len": max_len,
    }
    values.update({key: value for key, value in provided.items() if key in valid_fields})
    if bool(values.get("deterministic_derived_channels", False)):
        values.setdefault("target_feature_names", tuple(provided.get("target_feature_columns", ())))
        values.setdefault("base_feature_names", tuple(provided.get("base_output_feature_columns", ())))
    for key in (
        "target_feature_names",
        "base_feature_names",
        "output_feature_mean",
        "output_feature_std",
        "metadata_feature_mean",
        "metadata_feature_std",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return WearableCpetStudentModel(WearableStudentConfig(**values))


def _masked_mse_optional(pred: torch.Tensor | None, target: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
    if pred is None or target is None:
        reference = pred if pred is not None else mask
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    pred, target = _align_last_dim(pred, target)
    weights = mask.unsqueeze(-1).to(pred.dtype)
    error = (pred - target.to(pred.dtype)).pow(2) * weights
    return error.sum() / weights.sum().clamp_min(1.0) / pred.shape[-1]


def _trajectory_distillation(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    use_observability_weights: bool = False,
) -> torch.Tensor:
    target = batch.get("teacher_trajectory_mu", batch.get("trajectory_mu"))
    if target is None:
        return torch.zeros((), device=outputs["trajectory_mu"].device, dtype=outputs["trajectory_mu"].dtype)
    mu, target = _align_last_dim(outputs["trajectory_mu"], target)
    indices = _loss_feature_indices(batch, "distillation_feature_indices", mu.shape[-1], mu.device)
    if indices.numel() > 0:
        mu = mu.index_select(-1, indices)
        target = target.index_select(-1, indices)
    logvar = outputs["trajectory_logvar"][..., : mu.shape[-1]]
    if indices.numel() > 0:
        logvar = outputs["trajectory_logvar"].index_select(-1, indices)
    weights = mask.unsqueeze(-1).to(mu.dtype)
    if use_observability_weights:
        observability = batch.get("teacher_observability_weight")
        if observability is None:
            raise ValueError("observability-weighted distillation requires package weights")
        if observability.shape != batch["teacher_trajectory_mu"].shape:
            raise ValueError("teacher observability weight shape does not match trajectory")
        if indices.numel() > 0:
            observability = observability.index_select(-1, indices)
        weights = weights * observability.to(mu.dtype).clamp(0.0, 2.0)
    nll = 0.5 * (logvar + (target.to(mu.dtype) - mu).pow(2) / logvar.exp().clamp_min(1e-5))
    if weights.shape[-1] == 1:
        weights = weights.expand_as(nll)
    return (nll * weights).sum() / weights.sum().clamp_min(1.0)


def _teacher_logvar_target(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "teacher_trajectory_logvar" in batch:
        return batch["teacher_trajectory_logvar"]
    for key in ("teacher_trajectory_uncertainty", "trajectory_uncertainty"):
        if key in batch:
            return batch[key].clamp_min(1e-5).log()
    return None


def _selected_mse_optional(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    if pred is None or target is None:
        reference = pred if pred is not None else mask
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    pred, target = _align_last_dim(pred, target)
    indices = _loss_feature_indices(batch, key, pred.shape[-1], pred.device)
    if indices.numel() > 0:
        pred = pred.index_select(-1, indices)
        target = target.index_select(-1, indices)
    weights = mask.unsqueeze(-1).to(pred.dtype)
    error = (pred - target.to(pred.dtype)).pow(2) * weights
    return error.sum() / weights.sum().clamp_min(1.0) / pred.shape[-1]


def _selected_trajectory_loss(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    key: str,
) -> torch.Tensor:
    target = batch.get("target_trajectory")
    if target is None:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    mu, target = _align_last_dim(mu, target.to(mu.dtype))
    indices = _loss_feature_indices(batch, key, mu.shape[-1], mu.device)
    if indices.numel() == 0:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    selected_mask = mask
    channel_mask = batch.get("target_channel_mask")
    if channel_mask is not None:
        selected_mask = channel_mask.index_select(-1, indices).to(mask.dtype) * mask.unsqueeze(-1)
    return _masked_smooth_l1(
        mu.index_select(-1, indices), target.index_select(-1, indices), selected_mask
    )


def _endpoint_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    target_source: str = "auto",
) -> torch.Tensor:
    if target_source not in {"auto", "measured", "teacher"}:
        raise ValueError(f"unsupported endpoint target_source: {target_source!r}")
    losses = []
    if target_source == "teacher":
        at_target = batch.get("teacher_at_norm")
    elif target_source == "measured":
        at_target = batch.get("at_norm")
    else:
        at_target = batch.get("at_norm", batch.get("teacher_at_norm"))
    if at_target is not None:
        use_measured_at = target_source != "teacher" and "at_norm" in batch
        at_mask_key = "at_supervision_mask" if use_measured_at else "teacher_at_mask"
        at_mask = batch.get(at_mask_key, torch.ones_like(at_target)).to(outputs["at_norm"].dtype)
        at_valid = (at_mask > 0) & torch.isfinite(at_target) & (at_target >= 0) & (at_target <= 1)
        if bool(at_valid.any()):
            at_delta = F.smooth_l1_loss(
                outputs["at_norm"][at_valid],
                at_target.to(outputs["at_norm"].dtype)[at_valid],
                reduction="none",
            )
            losses.append((at_delta * at_mask[at_valid]).sum() / at_mask[at_valid].sum().clamp_min(1.0))
    measured_vo2_target = batch.get("vo2peak") if target_source != "teacher" else None
    if target_source == "teacher":
        vo2_target = batch.get("teacher_vo2peak")
    elif target_source == "measured":
        vo2_target = measured_vo2_target
    else:
        vo2_target = measured_vo2_target if measured_vo2_target is not None else batch.get("teacher_vo2peak")
    if vo2_target is not None:
        use_measured_mask = measured_vo2_target is not None and target_source != "teacher"
        mask_key = "vo2peak_mask" if use_measured_mask else "teacher_vo2peak_mask"
        vo2_mask = batch.get(mask_key, torch.ones_like(vo2_target)).to(outputs["vo2peak"].dtype)
        valid = (vo2_mask > 0) & torch.isfinite(vo2_target)
        if bool(valid.any()):
            scale = vo2_target[valid].abs().mean().clamp_min(1.0)
            delta = F.smooth_l1_loss(
                outputs["vo2peak"][valid] / scale,
                vo2_target.to(outputs["vo2peak"].dtype)[valid] / scale,
                reduction="none",
            )
            losses.append((delta * vo2_mask[valid]).sum() / vo2_mask[valid].sum().clamp_min(1.0))
    if not losses:
        references = [outputs["trajectory_mu"]]
        references.extend(outputs[key] for key in ("at_norm", "vo2peak") if key in outputs)
        return sum(_empty_loss(reference) for reference in references)
    return torch.stack(losses).sum()


def _empty_loss(reference: torch.Tensor) -> torch.Tensor:
    """Connected zero without evaluating possibly masked nonfinite values."""
    return reference.reshape(-1)[:0].sum()


def _teacher_at_supervision(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    extra_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target = batch.get("teacher_at_norm")
    if target is None or "at_norm" not in outputs:
        return _empty_loss(outputs["trajectory_mu"])
    mask = batch.get("teacher_at_mask", torch.ones_like(target)).to(outputs["at_norm"].dtype)
    if extra_mask is not None:
        mask = mask * extra_mask.to(mask.dtype)
    valid = (mask > 0) & torch.isfinite(target) & (target >= 0) & (target <= 1)
    if not bool(valid.any()):
        return _empty_loss(outputs["at_norm"])
    error = F.smooth_l1_loss(
        outputs["at_norm"][valid], target.to(outputs["at_norm"].dtype)[valid], reduction="none"
    )
    return (error * mask[valid]).sum() / mask[valid].sum().clamp_min(1.0)


def _distillation_relation_indices(
    batch: dict[str, torch.Tensor], width: int, device: torch.device
) -> torch.Tensor:
    return _loss_feature_indices(
        batch, "distillation_relation_feature_indices", width, device
    )


def _at_region_distillation(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    selection_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher = batch.get("teacher_trajectory_mu")
    teacher_at = batch.get("teacher_at_norm")
    if teacher is None or teacher_at is None:
        return _empty_loss(outputs["trajectory_mu"])
    student, teacher = _align_last_dim(outputs["trajectory_mu"], teacher)
    indices = _distillation_relation_indices(batch, student.shape[-1], student.device)
    if indices.numel() == 0:
        return _empty_loss(student)
    student = student.index_select(-1, indices)
    teacher = teacher.to(student.dtype).index_select(-1, indices)
    positions = _physical_time_fraction(batch, student, mask)
    bandwidth_raw = batch.get("distillation_at_bandwidth", torch.tensor(0.10, device=student.device))
    bandwidth = torch.as_tensor(bandwidth_raw, device=student.device, dtype=student.dtype).flatten()[0].clamp_min(1e-3)
    teacher_at_mask = batch.get("teacher_at_mask", torch.ones_like(teacher_at)).to(student.dtype)
    valid_at = (teacher_at_mask > 0) & torch.isfinite(teacher_at) & (teacher_at >= 0) & (teacher_at <= 1)
    centre = torch.where(valid_at, teacher_at, torch.zeros_like(teacher_at)).to(student.dtype).reshape(-1, 1)
    local_weight = torch.exp(-0.5 * ((positions - centre) / bandwidth).square())
    weights = mask.to(student.dtype).unsqueeze(-1) * local_weight.unsqueeze(-1)
    weights = weights * torch.where(valid_at, teacher_at_mask, torch.zeros_like(teacher_at_mask)).reshape(-1, 1, 1)
    teacher_logvar = batch.get("teacher_trajectory_logvar")
    if teacher_logvar is not None:
        teacher_logvar = _align_last_dim(outputs["trajectory_mu"], teacher_logvar)[1]
        confidence = torch.exp(-teacher_logvar.to(student.dtype).index_select(-1, indices).clamp(-4.0, 4.0))
        weights = weights * confidence
    if selection_mask is not None:
        selected = selection_mask.to(student.dtype)
        if selected.shape[-1] != student.shape[-1]:
            selected = selected.index_select(-1, indices)
        weights = weights * selected
    channel_mask = batch.get("target_channel_mask")
    if channel_mask is not None:
        weights = weights * channel_mask.to(student.dtype).index_select(-1, indices)
    finite = torch.isfinite(student) & torch.isfinite(teacher)
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    error = F.smooth_l1_loss(
        torch.where(finite, student, torch.zeros_like(student)),
        torch.where(finite, teacher, torch.zeros_like(teacher)),
        reduction="none",
    )
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def _relation_distillation(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    selection_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher = batch.get("teacher_trajectory_mu")
    if teacher is None:
        return _empty_loss(outputs["trajectory_mu"])
    student, teacher = _align_last_dim(outputs["trajectory_mu"], teacher)
    indices = _distillation_relation_indices(batch, student.shape[-1], student.device)
    if indices.numel() == 0 or student.shape[1] < 2:
        return _empty_loss(student)
    student = student.index_select(-1, indices)
    teacher = teacher.to(student.dtype).index_select(-1, indices)
    adjacent_mask = (mask[:, 1:] * mask[:, :-1]).to(student.dtype).unsqueeze(-1)
    if selection_mask is not None:
        selected = selection_mask.to(student.dtype)
        if selected.shape[-1] != student.shape[-1]:
            selected = selected.index_select(-1, indices)
        adjacent_mask = adjacent_mask * selected[:, 1:] * selected[:, :-1]
    student_delta = student[:, 1:] - student[:, :-1]
    teacher_delta = teacher[:, 1:] - teacher[:, :-1]
    slope_error = F.smooth_l1_loss(student_delta, teacher_delta, reduction="none")
    slope_loss = (slope_error * adjacent_mask).sum() / adjacent_mask.sum().clamp_min(1.0)

    pair_losses = []
    for left in range(student.shape[-1]):
        for right in range(left + 1, student.shape[-1]):
            student_relation = student[..., left] - student[..., right]
            teacher_relation = teacher[..., left] - teacher[..., right]
            pair_weight = mask.to(student.dtype)
            if selection_mask is not None:
                selected = selection_mask.to(student.dtype)
                if selected.shape[-1] != student.shape[-1]:
                    selected = selected.index_select(-1, indices)
                pair_weight = pair_weight * selected[..., left] * selected[..., right]
            pair_error = F.smooth_l1_loss(student_relation, teacher_relation, reduction="none")
            pair_losses.append((pair_error * pair_weight).sum() / pair_weight.sum().clamp_min(1.0))
    cross_channel_loss = torch.stack(pair_losses).mean() if pair_losses else _empty_loss(student)

    student_latent = outputs.get("latent_rollout")
    teacher_latent = batch.get("teacher_latent_rollout")
    latent_loss = _empty_loss(student)
    if student_latent is not None and teacher_latent is not None and student_latent.shape[1] >= 2:
        student_latent, teacher_latent = _align_last_dim(student_latent, teacher_latent)
        student_change = student_latent[:, 1:] - student_latent[:, :-1]
        teacher_change = teacher_latent.to(student_latent.dtype)[:, 1:] - teacher_latent.to(student_latent.dtype)[:, :-1]
        cosine_error = 1.0 - F.cosine_similarity(student_change, teacher_change, dim=-1, eps=1e-6)
        latent_weight = adjacent_mask.mean(dim=-1)
        latent_loss = (cosine_error * latent_weight).sum() / latent_weight.sum().clamp_min(1.0)
    return (slope_loss + cross_channel_loss + latent_loss) / 3.0


def _teacher_morphology_distillation(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    teacher = batch.get("teacher_trajectory_mu")
    if teacher is None:
        return _empty_loss(outputs["trajectory_mu"])
    student, teacher = _align_last_dim(outputs["trajectory_mu"], teacher)
    indices = _distillation_relation_indices(batch, student.shape[-1], student.device)
    if indices.numel() == 0:
        return _empty_loss(student)
    student = student.index_select(-1, indices)
    teacher = teacher.to(student.dtype).index_select(-1, indices)
    weights = mask.to(student.dtype).unsqueeze(-1).expand_as(student)
    finite = torch.isfinite(student) & torch.isfinite(teacher)
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    safe_student = torch.where(finite, student, torch.zeros_like(student))
    safe_teacher = torch.where(finite, teacher, torch.zeros_like(teacher))
    student_mean = (safe_student * weights).sum(dim=1, keepdim=True) / count
    teacher_mean = (safe_teacher * weights).sum(dim=1, keepdim=True) / count
    student_centered = safe_student - student_mean
    teacher_centered = safe_teacher - teacher_mean
    student_scale = torch.sqrt(
        (student_centered.square() * weights).sum(dim=1, keepdim=True) / count
    ).clamp_min(1e-4)
    teacher_scale = torch.sqrt(
        (teacher_centered.square() * weights).sum(dim=1, keepdim=True) / count
    ).clamp_min(1e-4)
    student_shape = student_centered / student_scale
    teacher_shape = teacher_centered / teacher_scale
    error = F.smooth_l1_loss(student_shape, teacher_shape, reduction="none")
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def _selective_teacher_distillation(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], mask: torch.Tensor
) -> torch.Tensor:
    target = batch.get("target_trajectory")
    teacher = batch.get("teacher_trajectory_mu")
    reference = batch.get("reference_trajectory_mu")
    if target is None or teacher is None or reference is None:
        return _empty_loss(outputs["trajectory_mu"])
    student, target = _align_last_dim(outputs["trajectory_mu"], target)
    _, teacher = _align_last_dim(student, teacher)
    _, reference = _align_last_dim(student, reference)
    indices = _distillation_relation_indices(batch, student.shape[-1], student.device)
    if indices.numel() == 0:
        return _empty_loss(student)
    target_selected = target.to(student.dtype).index_select(-1, indices)
    teacher_selected = teacher.to(student.dtype).index_select(-1, indices)
    reference_selected = reference.to(student.dtype).index_select(-1, indices)
    margin_raw = batch.get("selective_distillation_margin", torch.tensor(0.0, device=student.device))
    margin = torch.as_tensor(margin_raw, device=student.device, dtype=student.dtype).flatten()[0]
    teacher_error = (teacher_selected - target_selected).square()
    reference_error = (reference_selected - target_selected).square()
    gate = (teacher_error + margin < reference_error).to(student.dtype)
    channel_mask = batch.get("target_channel_mask")
    if channel_mask is not None:
        gate = gate * channel_mask.to(student.dtype).index_select(-1, indices)
    gate = gate * mask.to(student.dtype).unsqueeze(-1)

    at_gate = None
    measured_at = batch.get("at_norm")
    teacher_at = batch.get("teacher_at_norm")
    reference_at = batch.get("reference_at_norm")
    if measured_at is not None and teacher_at is not None and reference_at is not None:
        teacher_at_error = (teacher_at.to(student.dtype) - measured_at.to(student.dtype)).abs()
        reference_at_error = (reference_at.to(student.dtype) - measured_at.to(student.dtype)).abs()
        at_gate = (teacher_at_error + margin < reference_at_error).to(student.dtype)
        at_gate = at_gate * batch.get("at_supervision_mask", torch.ones_like(at_gate)).to(student.dtype)

    at_loss = _teacher_at_supervision(outputs, batch, at_gate)
    local_loss = _at_region_distillation(outputs, batch, mask, gate)
    relation_loss = _relation_distillation(outputs, batch, mask, gate)
    at_weight = float(torch.as_tensor(batch.get("selective_at_weight", 1.0)).flatten()[0].item())
    local_weight = float(torch.as_tensor(batch.get("selective_at_region_weight", 0.25)).flatten()[0].item())
    relation_weight = float(torch.as_tensor(batch.get("selective_relation_weight", 0.25)).flatten()[0].item())
    return at_weight * at_loss + local_weight * local_loss + relation_weight * relation_loss


def _physiology_consistency(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    mean = batch.get("target_feature_mean")
    std = batch.get("target_feature_std")
    if mean is None or std is None:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    mean = mean.to(device=mu.device, dtype=mu.dtype).unsqueeze(1)
    std = std.to(device=mu.device, dtype=mu.dtype).clamp_min(1e-6).unsqueeze(1)
    if mean.shape[-1] != mu.shape[-1] or std.shape[-1] != mu.shape[-1]:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    physical = mu * std + mean
    weights = mask.unsqueeze(-1).to(mu.dtype)
    # Normalize violations by the within-exam scale so heterogeneous physical
    # units contribute comparably. Do not penalize temporal derivatives here;
    # measured shape is supervised explicitly below.
    negative_z = F.relu(-physical) / std
    return (negative_z * weights).sum() / weights.sum().clamp_min(1.0) / mu.shape[-1]


def _derived_consistency(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Enforce CPET algebra after restoring physical units.

    ``derived_feature_indices`` is ordered as VO2, VO2_kg, VCO2, VCO2_kg,
    VE, RER, VE_VO2, VE_VCO2, METS, HR, and VO2_HR.
    """
    mean = batch.get("target_feature_mean")
    std = batch.get("target_feature_std")
    raw_indices = batch.get("derived_feature_indices")
    body_mass = batch.get("body_mass_kg")
    if mean is None or std is None or raw_indices is None or body_mass is None:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    indices = raw_indices.to(device=mu.device, dtype=torch.long).flatten()
    if indices.numel() != 11 or bool((indices < 0).any()) or bool((indices >= mu.shape[-1]).any()):
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    mean = mean.to(device=mu.device, dtype=mu.dtype)
    std = std.to(device=mu.device, dtype=mu.dtype).clamp_min(1e-6)
    if mean.shape[-1] != mu.shape[-1] or std.shape[-1] != mu.shape[-1]:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    physical = mu * std.unsqueeze(1) + mean.unsqueeze(1)
    vo2, vo2_kg, vco2, vco2_kg, ve, rer, ve_vo2, ve_vco2, mets, hr, vo2_hr = [
        physical[..., int(index)] for index in indices
    ]
    raw_mass = body_mass.to(device=mu.device, dtype=mu.dtype).reshape(-1, 1)
    mass_valid = ((raw_mass >= 20.0) & (raw_mass <= 300.0)).to(mu.dtype)
    mass = torch.where(mass_valid > 0, raw_mass, torch.ones_like(raw_mass))
    expected_physical = (
        vo2 / mass,
        vco2 / mass,
        vco2 / vo2.clamp_min(1.0),
        1000.0 * ve / vo2.clamp_min(1.0),
        1000.0 * ve / vco2.clamp_min(1.0),
        vo2_kg / 3.5,
        vo2 / hr.clamp_min(1.0),
    )
    predicted = (vo2_kg, vco2_kg, rer, ve_vo2, ve_vco2, mets, vo2_hr)
    derived_indices = (indices[1], indices[3], indices[5], indices[6], indices[7], indices[8], indices[10])
    consistency_masks = (mask * mass_valid, mask * mass_valid, mask, mask, mask, mask, mask)
    losses = []
    for predicted_physical, expected, channel_index, consistency_mask in zip(
        predicted, expected_physical, derived_indices, consistency_masks
    ):
        channel_std = std[:, int(channel_index)].unsqueeze(1).clamp_min(1e-6)
        predicted_z = predicted_physical / channel_std
        expected_z = expected / channel_std
        losses.append(
            _masked_smooth_l1(predicted_z.unsqueeze(-1), expected_z.unsqueeze(-1), consistency_mask)
        )
    return torch.stack(losses).mean()


def _measured_curve_losses(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch.get("target_trajectory")
    direct_indices = _loss_feature_indices(batch, "direct_loss_feature_indices", mu.shape[-1], mu.device)
    peak_indices = _loss_feature_indices(batch, "peak_loss_feature_indices", mu.shape[-1], mu.device)
    slope_indices = _loss_feature_indices(batch, "slope_loss_feature_indices", mu.shape[-1], mu.device)
    if target is None or direct_indices.numel() == 0:
        zero = torch.zeros((), device=mu.device, dtype=mu.dtype)
        return zero, zero, zero
    mu, target = _align_last_dim(mu, target.to(mu.dtype))
    shape_mask = batch.get("shape_mask", batch.get("future_mask", mask)).to(mu.dtype)
    if shape_mask.shape != mask.shape:
        shape_mask = mask
    else:
        shape_mask = shape_mask * mask
    direct_mask = shape_mask
    channel_mask = batch.get("target_channel_mask")
    if channel_mask is not None:
        direct_mask = channel_mask.index_select(-1, direct_indices).to(shape_mask.dtype) * shape_mask.unsqueeze(-1)
    direct = _masked_smooth_l1(
        mu.index_select(-1, direct_indices), target.index_select(-1, direct_indices), direct_mask
    )
    zero = torch.zeros((), device=mu.device, dtype=mu.dtype)
    peak = zero
    if peak_indices.numel() > 0:
        peak = _future_peak_loss(
            mu.index_select(-1, peak_indices), target.index_select(-1, peak_indices), shape_mask,
            channel_mask.index_select(-1, peak_indices) if channel_mask is not None else None,
        )
    slope = zero
    if slope_indices.numel() > 0:
        slope = _future_slope_loss(
            mu.index_select(-1, slope_indices), target.index_select(-1, slope_indices), mask, shape_mask,
            channel_mask.index_select(-1, slope_indices) if channel_mask is not None else None,
        )
    return direct, peak, slope


def _boundary_morphology_timing_losses(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch.get("target_trajectory")
    direct_indices = _loss_feature_indices(batch, "direct_loss_feature_indices", mu.shape[-1], mu.device)
    peak_indices = _loss_feature_indices(batch, "peak_loss_feature_indices", mu.shape[-1], mu.device)
    zero = torch.zeros((), device=mu.device, dtype=mu.dtype)
    if target is None or direct_indices.numel() == 0:
        return zero, zero, zero
    mu, target = _align_last_dim(mu, target.to(mu.dtype))
    channel_mask = batch.get("target_channel_mask")
    boundary_steps = int(batch.get("initial_boundary_steps", torch.tensor(8)).flatten()[0].item())
    boundary_steps = min(max(boundary_steps, 1), mu.shape[1])
    boundary_mask = mask.clone()
    boundary_mask[:, boundary_steps:] = 0
    if channel_mask is not None:
        boundary_mask = boundary_mask.unsqueeze(-1) * channel_mask.index_select(-1, direct_indices).to(mu.dtype)
    boundary = _masked_smooth_l1(
        mu.index_select(-1, direct_indices), target.index_select(-1, direct_indices), boundary_mask
    )

    shape_mask = batch.get("shape_mask", mask).to(mu.dtype) * mask
    direct_pred = mu.index_select(-1, direct_indices)
    direct_target = target.index_select(-1, direct_indices)
    weights = shape_mask.unsqueeze(-1)
    if channel_mask is not None:
        weights = weights * channel_mask.index_select(-1, direct_indices).to(mu.dtype)
    finite = torch.isfinite(direct_pred) & torch.isfinite(direct_target)
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    direct_pred = torch.where(finite, direct_pred, torch.zeros_like(direct_pred))
    direct_target = torch.where(finite, direct_target, torch.zeros_like(direct_target))
    count = weights.sum(dim=1).clamp_min(1.0)
    pred_centered = direct_pred - (direct_pred * weights).sum(dim=1, keepdim=True) / count.unsqueeze(1)
    target_centered = direct_target - (direct_target * weights).sum(dim=1, keepdim=True) / count.unsqueeze(1)
    numerator = (pred_centered * target_centered * weights).sum(dim=1)
    energy_product = (
        (pred_centered.square() * weights).sum(dim=1)
        * (target_centered.square() * weights).sum(dim=1)
    )
    denominator = energy_product.clamp_min(1e-8).sqrt()
    valid_corr = (count > 2) & (energy_product > 1e-8)
    correlation = (numerator / denominator.clamp_min(1e-6)).clamp(-1.0, 1.0)
    morphology = ((1.0 - correlation) * valid_corr).sum() / valid_corr.sum().clamp_min(1)

    peak_timing = zero
    if peak_indices.numel() > 0:
        peak_pred = mu.index_select(-1, peak_indices)
        peak_target = target.index_select(-1, peak_indices)
        peak_valid = shape_mask.unsqueeze(-1) > 0
        if channel_mask is not None:
            peak_valid = peak_valid & (channel_mask.index_select(-1, peak_indices) > 0)
        peak_valid = peak_valid & torch.isfinite(peak_pred) & torch.isfinite(peak_target)
        row_valid = peak_valid.sum(dim=1) > 2
        positions = _physical_time_fraction(batch, mu, mask).unsqueeze(-1)
        floor = torch.full_like(peak_pred, -1e4)
        beta = float(batch.get("peak_timing_beta", torch.tensor(10.0)).flatten()[0].item())
        pred_weights = torch.softmax(torch.where(peak_valid, peak_pred * beta, floor), dim=1)
        target_weights = torch.softmax(torch.where(peak_valid, peak_target * beta, floor), dim=1)
        pred_time = (pred_weights * positions).sum(dim=1)
        target_time = (target_weights * positions).sum(dim=1)
        timing_error = F.smooth_l1_loss(pred_time, target_time, reduction="none") * row_valid.to(mu.dtype)
        peak_timing = timing_error.sum() / row_valid.sum().clamp_min(1)
    return boundary, morphology, peak_timing


def _valid_measured_at(batch: dict[str, torch.Tensor], reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = batch["at_norm"].to(device=reference.device, dtype=reference.dtype).reshape(-1)
    observed = batch.get("at_supervision_mask", torch.ones_like(raw)).to(device=reference.device).reshape(-1)
    valid = (observed > 0) & torch.isfinite(raw) & (raw >= 0) & (raw <= 1)
    return torch.where(valid, raw, torch.zeros_like(raw)), valid.to(reference.dtype)


def _physical_time_fraction(
    batch: dict[str, torch.Tensor], reference: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Coordinates share the source-time origin/span used by AT labels.

    Historical callers without explicit times get a valid-row-rank fallback,
    not the padded index axis. New v5 batches always carry physical times.
    """
    valid = mask.to(device=reference.device) > 0
    positions = batch.get("time_fraction")
    if positions is not None:
        positions = positions.to(device=reference.device, dtype=torch.float32)
        if positions.shape != valid.shape:
            raise ValueError("time_fraction must align with the trajectory time mask")
        if not bool(torch.isfinite(positions[valid]).all()):
            raise ValueError("valid observations require finite physical time coordinates")
        return torch.where(valid, positions, torch.zeros_like(positions))
    ranks = valid.to(torch.float32).cumsum(dim=1) - 1
    span = (valid.sum(dim=1, keepdim=True) - 1).clamp_min(1)
    return torch.where(valid, ranks / span, torch.zeros_like(ranks))


def _core_derived_curve_losses(
    mu: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Supervise selected deterministic ratios without treating them as measured outputs."""
    zero = torch.zeros((), device=mu.device, dtype=mu.dtype)
    target = batch.get("target_trajectory")
    indices = _loss_feature_indices(batch, "core_derived_feature_indices", mu.shape[-1], mu.device)
    if target is None or indices.numel() == 0:
        return zero, zero, zero, zero, zero

    mu, target = _align_last_dim(mu, target.to(mu.dtype))
    pred = mu.index_select(-1, indices)
    expected = target.index_select(-1, indices)
    shape_mask = batch.get("shape_mask", mask).to(mu.dtype)
    if shape_mask.shape != mask.shape:
        shape_mask = mask
    else:
        shape_mask = shape_mask * mask
    weights = shape_mask.unsqueeze(-1)
    channel_mask = batch.get("target_channel_mask")
    if channel_mask is not None:
        weights = weights * channel_mask.index_select(-1, indices).to(mu.dtype)
    confidence = batch.get("core_derived_confidence")
    if confidence is not None:
        confidence = confidence.to(mu.dtype)
        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        weights = weights * confidence

    positions = _physical_time_fraction(batch, mu, mask).unsqueeze(-1)
    at_norm = batch.get("at_norm")
    if at_norm is not None:
        at, at_valid = _valid_measured_at(batch, mu)
        at = at.reshape(-1, 1, 1)
        bandwidth = max(float(batch.get("core_at_bandwidth", torch.tensor(0.12)).flatten()[0].item()), 1e-3)
        at_weight = float(batch.get("core_at_weight", torch.tensor(1.0)).flatten()[0].item())
        post_at_weight = float(batch.get("core_post_at_weight", torch.tensor(0.5)).flatten()[0].item())
        at_focus = torch.exp(-0.5 * ((positions - at) / bandwidth).square())
        post_at = (positions >= at).to(mu.dtype)
        weights = weights * (1.0 + at_valid.reshape(-1, 1, 1) * (at_weight * at_focus + post_at_weight * post_at))

    trajectory = _masked_smooth_l1(pred, expected, weights)
    finite = torch.isfinite(pred) & torch.isfinite(expected)
    safe_weights = torch.where(finite, weights, torch.zeros_like(weights))
    safe_pred = torch.where(finite, pred, torch.zeros_like(pred))
    safe_expected = torch.where(finite, expected, torch.zeros_like(expected))

    count = safe_weights.sum(dim=1).clamp_min(1.0)
    pred_centered = safe_pred - (safe_pred * safe_weights).sum(dim=1, keepdim=True) / count.unsqueeze(1)
    expected_centered = safe_expected - (
        (safe_expected * safe_weights).sum(dim=1, keepdim=True) / count.unsqueeze(1)
    )
    numerator = (pred_centered * expected_centered * safe_weights).sum(dim=1)
    energy_product = (
        (pred_centered.square() * safe_weights).sum(dim=1)
        * (expected_centered.square() * safe_weights).sum(dim=1)
    )
    valid_corr = (count > 2) & (energy_product > 1e-8)
    correlation = (numerator / energy_product.clamp_min(1e-8).sqrt()).clamp(-1.0, 1.0)
    morphology = ((1.0 - correlation) * valid_corr).sum() / valid_corr.sum().clamp_min(1)

    if mu.shape[1] > 1:
        slope_weights = torch.minimum(safe_weights[:, 1:], safe_weights[:, :-1])
        slope = _masked_smooth_l1(
            safe_pred[:, 1:] - safe_pred[:, :-1],
            safe_expected[:, 1:] - safe_expected[:, :-1],
            slope_weights,
        )
    else:
        slope = zero

    peak_valid = safe_weights > 0
    row_valid = peak_valid.sum(dim=1) > 2
    floor = torch.full_like(pred, -1e4)
    masked_pred = torch.where(peak_valid, pred, floor)
    masked_expected = torch.where(peak_valid, expected, floor)
    peak_error = F.smooth_l1_loss(masked_pred.max(dim=1).values, masked_expected.max(dim=1).values, reduction="none")
    peak = (peak_error * row_valid.to(mu.dtype)).sum() / row_valid.sum().clamp_min(1)

    beta = float(batch.get("peak_timing_beta", torch.tensor(10.0)).flatten()[0].item())
    log_confidence = safe_weights.clamp_min(1e-8).log()
    pred_peak_weights = torch.softmax(torch.where(peak_valid, pred * beta + log_confidence, floor), dim=1)
    expected_peak_weights = torch.softmax(
        torch.where(peak_valid, expected * beta + log_confidence, floor), dim=1
    )
    pred_time = (pred_peak_weights * positions).sum(dim=1)
    expected_time = (expected_peak_weights * positions).sum(dim=1)
    timing_error = F.smooth_l1_loss(pred_time, expected_time, reduction="none")
    peak_timing = (timing_error * row_valid.to(mu.dtype)).sum() / row_valid.sum().clamp_min(1)
    return trajectory, morphology, slope, peak, peak_timing


def _loss_feature_indices(
    batch: dict[str, torch.Tensor],
    key: str,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    raw = batch.get(key, batch.get("shape_feature_indices"))
    if raw is None:
        return torch.empty(0, device=device, dtype=torch.long)
    indices = raw.to(device=device, dtype=torch.long).flatten()
    return indices[(indices >= 0) & (indices < width)]


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(pred.dtype) if mask.ndim == pred.ndim else mask.unsqueeze(-1).to(pred.dtype)
    finite = torch.isfinite(pred) & torch.isfinite(target)
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    safe_pred = torch.where(finite, pred, torch.zeros_like(pred))
    safe_target = torch.where(finite, target, torch.zeros_like(target))
    error = F.smooth_l1_loss(safe_pred, safe_target, reduction="none") * weights
    denominator = weights.sum().clamp_min(1.0)
    if weights.shape[-1] == 1 and pred.shape[-1] != 1:
        denominator = denominator * pred.shape[-1]
    return error.sum() / denominator


def _future_peak_loss(
    pred: torch.Tensor, target: torch.Tensor, future_mask: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = (future_mask.unsqueeze(-1) > 0) & torch.isfinite(pred) & torch.isfinite(target)
    if channel_mask is not None:
        valid = valid & (channel_mask > 0)
    valid_channels = valid.any(dim=1)
    floor = torch.full_like(pred, torch.finfo(pred.dtype).min)
    pred_peak = torch.where(valid, pred, floor).max(dim=1).values
    target_peak = torch.where(valid, target, floor).max(dim=1).values
    return _masked_smooth_l1(pred_peak, target_peak, valid_channels)


def _future_slope_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    future_mask: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    valid = (mask.unsqueeze(-1) > 0) & torch.isfinite(pred) & torch.isfinite(target)
    if channel_mask is not None:
        valid = valid & (channel_mask > 0)
    safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    pred_delta = safe_pred[:, 1:, :] - safe_pred[:, :-1, :]
    target_delta = safe_target[:, 1:, :] - safe_target[:, :-1, :]
    slope_mask = valid[:, 1:] & valid[:, :-1] & (future_mask[:, 1:, None] > 0)
    return _masked_smooth_l1(pred_delta, target_delta, slope_mask)


def _align_last_dim(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = min(left.shape[-1], right.shape[-1])
    return left[..., :width], right[..., :width]
