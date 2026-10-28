from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(target)
    if np.any(mask & ~np.isfinite(pred)):
        raise FloatingPointError(f"Non-finite prediction at eligible {prefix} target")
    if mask.sum() == 0:
        return {f"{prefix}_mae": float("nan"), f"{prefix}_bias": float("nan"), f"{prefix}_rmse": float("nan")}
    delta = pred[mask] - target[mask]
    return {
        f"{prefix}_mae": float(np.mean(np.abs(delta))),
        f"{prefix}_bias": float(np.mean(delta)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(delta**2))),
    }


def at_time_metrics(
    pred_norm: np.ndarray,
    target_seconds: np.ndarray,
    time_start: np.ndarray,
    duration: np.ndarray,
    eligibility: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return AT timing error in seconds from normalized model predictions."""
    pred_norm = np.asarray(pred_norm, dtype=float)
    target_seconds = np.asarray(target_seconds, dtype=float)
    time_start = np.asarray(time_start, dtype=float)
    duration = np.asarray(duration, dtype=float)
    pred_seconds = time_start + pred_norm * duration
    valid = (
        np.isfinite(time_start)
        & np.isfinite(target_seconds)
        & np.isfinite(duration)
        & (duration > 0)
    )
    if eligibility is not None:
        eligibility = np.asarray(eligibility, dtype=float)
        if eligibility.shape != valid.shape:
            raise ValueError(
                f"AT eligibility shape {eligibility.shape} does not match endpoint shape {valid.shape}"
            )
        valid &= np.isfinite(eligibility) & (eligibility > 0)
    metrics: dict[str, float | int] = regression_metrics(
        pred_seconds[valid], target_seconds[valid], "at_time_s"
    )
    metrics["at_time_s_evaluable_count"] = int(valid.sum())
    return metrics


def trajectory_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    weights = np.asarray(mask, dtype=float)
    if weights.ndim == pred.ndim - 1:
        weights = weights[..., None]
    weights = np.broadcast_to(weights, pred.shape)
    valid = (weights > 0) & np.isfinite(target)
    if np.any(valid & ~np.isfinite(pred)):
        raise FloatingPointError("Non-finite prediction at an eligible trajectory target")
    if not valid.any():
        return {"trajectory_rmse": float("nan")}
    error = pred[valid] - target[valid]
    return {"trajectory_rmse": float(np.sqrt(np.average(error**2, weights=weights[valid])))}
