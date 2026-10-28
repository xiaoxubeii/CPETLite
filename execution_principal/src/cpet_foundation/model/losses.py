from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(pred.dtype)
    error = (pred - target).pow(2) * weights
    return error.sum() / weights.sum().clamp_min(1.0) / pred.shape[-1]


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = (mask.unsqueeze(-1) if mask.ndim == 2 else mask).to(mu.dtype)
    loss = 0.5 * (logvar + (target - mu).pow(2) / logvar.exp().clamp_min(1e-5))
    if weights.shape[-1] == 1:
        weights = weights.expand_as(loss)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def endpoint_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    at_mask = batch.get("at_supervision_mask", torch.ones_like(batch["at_norm"])).to(
        outputs["at_norm"].dtype
    )
    at_delta = F.smooth_l1_loss(
        outputs["at_norm"], batch["at_norm"].to(outputs["at_norm"].dtype), reduction="none"
    )
    at_loss = (at_delta * at_mask).sum() / at_mask.sum().clamp_min(1.0)
    vo2_mask = batch.get("vo2peak_mask", torch.ones_like(batch["vo2peak"])).to(outputs["vo2peak"].dtype)
    valid_vo2 = vo2_mask > 0
    if bool(valid_vo2.any()):
        vo2_scale = batch["vo2peak"][valid_vo2].abs().mean().clamp_min(1.0)
        vo2_delta = F.smooth_l1_loss(
            outputs["vo2peak"] / vo2_scale,
            batch["vo2peak"] / vo2_scale,
            reduction="none",
        )
        vo2_loss = (vo2_delta * vo2_mask).sum() / vo2_mask.sum().clamp_min(1.0)
    else:
        vo2_loss = torch.zeros((), device=outputs["vo2peak"].device, dtype=outputs["vo2peak"].dtype)
    return at_loss + 0.25 * vo2_loss


def physiology_graph_loss(outputs: dict[str, torch.Tensor], feature_index: dict[str, int]) -> torch.Tensor:
    mu = outputs["trajectory_mu"]
    losses = []
    for name in ("VO2", "VCO2", "VE", "RER", "HR"):
        if name in feature_index:
            losses.append(F.relu(-mu[..., feature_index[name]]).mean())
    if "Power_Load" in feature_index and "VO2" in feature_index and mu.shape[1] > 2:
        d_work = mu[:, 1:, feature_index["Power_Load"]] - mu[:, :-1, feature_index["Power_Load"]]
        d_vo2 = mu[:, 1:, feature_index["VO2"]] - mu[:, :-1, feature_index["VO2"]]
        losses.append(F.relu(d_work * -d_vo2).mean() * 0.05)
    if not losses:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    return torch.stack(losses).sum()


def total_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    feature_index: dict[str, int],
    reconstruction_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    trajectory_mask = batch["mask"].to(outputs["trajectory_mu"].dtype)
    sample_mask = batch.get("trajectory_supervision_mask")
    if sample_mask is not None:
        trajectory_mask = trajectory_mask * sample_mask.to(trajectory_mask.dtype).reshape(-1, 1)
    if "feature_mask" in batch:
        trajectory_mask = trajectory_mask.unsqueeze(-1) * batch["feature_mask"].to(trajectory_mask.dtype)
    recon = gaussian_nll(
        outputs["trajectory_mu"],
        outputs["trajectory_logvar"],
        batch["x"],
        trajectory_mask,
    )
    endpoint = endpoint_loss(outputs, batch)
    graph = physiology_graph_loss(outputs, feature_index)
    loss = reconstruction_weight * recon + endpoint + 0.1 * graph
    return loss, {
        "loss": float(loss.detach().cpu()),
        "reconstruction": float(recon.detach().cpu()),
        "endpoint": float(endpoint.detach().cpu()),
        "graph": float(graph.detach().cpu()),
    }
