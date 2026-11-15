"""BARN navigation metric (pure torch, no Isaac imports).

nav_metric = int(success) * optimal_time / clip(actual_time, 2*optimal_time, 8*optimal_time)
optimal_time = path_length / 2.0

The ceiling is 0.5, reached at an average speed >= 1 m/s along the planned path.
A failure (collision / timeout) always scores 0.
"""

from __future__ import annotations

import torch


def optimal_time(path_length: torch.Tensor | float, dynabarn_parity: bool = False) -> torch.Tensor:
    """optimal_time = path_length / 2.0.

    With ``dynabarn_parity`` the raw (negative) DynaBARN path length is used so
    the upstream degenerate clip is reproduced; otherwise the magnitude is used.
    """
    pl = torch.as_tensor(path_length, dtype=torch.float32)
    if dynabarn_parity:
        return pl / 2.0
    return torch.abs(pl) / 2.0


def nav_metric(
    success: torch.Tensor | bool,
    actual_time: torch.Tensor | float,
    optimal_time: torch.Tensor | float,
    dynabarn_parity: bool = False,
) -> torch.Tensor:
    """BARN navigation metric per episode.

    Args:
        success: (N,) bool/float — reached the goal within the threshold.
        actual_time: (N,) seconds elapsed.
        optimal_time: (N,) path_length / 2.0.
        dynabarn_parity: reproduce the upstream DynaBARN degenerate clip
            (every success scores exactly 0.125 regardless of time).
    Returns:
        (N,) metric in [0, 0.5]; zero on collision / timeout / failure.
    """
    success = success.to(torch.float32) if torch.is_tensor(success) else torch.tensor(
        float(success), dtype=torch.float32
    )
    actual_time = torch.as_tensor(actual_time, dtype=torch.float32)
    optimal_time = torch.as_tensor(optimal_time, dtype=torch.float32)

    if dynabarn_parity:
        # Upstream DynaBARN: path_length = -31 m -> clip bounds invert -> 0.125
        # per success for any actual_time.
        return success * torch.full_like(actual_time, 0.125)

    lower = 2.0 * optimal_time
    upper = 8.0 * optimal_time
    clipped = torch.clamp(actual_time, lower, upper)
    return success * optimal_time / clipped
