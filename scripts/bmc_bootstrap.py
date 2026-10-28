"""Conditional examination bootstrap shared by every fitted seed/model pair."""
import numpy as np


def paired_intervals(seed_pairs, metric_function, *, draws=2000, seed=42):
    n = len(seed_pairs[0][0]['core_sse'])
    assert n and all(len(a['core_sse']) == len(b['core_sse']) == n for a, b in seed_pairs)
    point = np.stack([metric_function(a) - metric_function(b) for a, b in seed_pairs]).mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        # Examinations recur across optimization seeds. A single draw preserves
        # this dependence while keeping the teacher and fitted students fixed.
        indices = rng.integers(0, n, size=n)
        differences = [metric_function(a, indices) - metric_function(b, indices) for a, b in seed_pairs]
        samples.append(np.stack(differences).mean(axis=0))
    low, high = np.quantile(np.asarray(samples), [.025, .975], axis=0)
    return np.stack([point, low, high])
