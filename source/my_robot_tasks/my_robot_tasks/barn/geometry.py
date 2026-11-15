"""Geometry primitives for BARN (pure torch, no Isaac imports).

* min-pooling of lidar ranges (contiguous bins, never stride slicing)
* analytic ray-vs-circle lidar (batched torch)
* oriented-rectangle vs circle footprint test (exact for BARN geometry)
"""

from __future__ import annotations

import torch


def min_pool_ranges(ranges: torch.Tensor, out_bins: int) -> torch.Tensor:
    """Min-pool contiguous bins of a 1D range array (never stride slicing).

    A 0.075 m cylinder at 3 m subtends ~2.9°, while 64 bins over 270° are 4.2°
    wide, so stride slicing can step over a cylinder entirely. Min-pooling over
    contiguous bins guarantees a hit on any single beam survives downsampling.

    Args:
        ranges: (..., N) input ranges (inf = no hit).
        out_bins: number of output bins.
    Returns:
        (..., out_bins) min-pooled ranges.
    """
    n = ranges.shape[-1]
    device = ranges.device
    # Assign each input beam to one output bin; bins are contiguous and cover [0, n).
    bins = (torch.arange(n, device=device, dtype=torch.float32) * out_bins / n).floor().long()
    bins = bins.clamp(max=out_bins - 1)
    out = torch.full(
        ranges.shape[:-1] + (out_bins,), float("inf"), device=device, dtype=ranges.dtype
    )
    index = bins.expand_as(ranges)
    out = out.scatter_reduce(-1, index, ranges, reduce="amin", include_self=True)
    return out


def analytic_lidar(
    origins: torch.Tensor,
    headings: torch.Tensor,
    beam_angles: torch.Tensor,
    cylinder_centers: torch.Tensor,
    radius: float,
    max_distance: float,
) -> torch.Tensor:
    """Analytic ray-vs-circle lidar, batched over envs and beams.

    Args:
        origins: (E, 2) ray origins [m].
        headings: (E,) robot headings [rad].
        beam_angles: (B,) beam angles relative to heading [rad].
        cylinder_centers: (C, 2) cylinder centers [m].
        radius: cylinder radius [m].
        max_distance: sensor max range [m].
    Returns:
        (E, B) distance to nearest hit (inf = no hit within max_distance).
    """
    dirs = torch.stack(
        (
            torch.cos(headings[:, None] + beam_angles[None, :]),
            torch.sin(headings[:, None] + beam_angles[None, :]),
        ),
        dim=-1,
    )  # (E, B, 2)

    # f = origin - center, broadcast (E, B, C, 2)
    f = origins[:, None, None, :] - cylinder_centers[None, None, :, :]
    df = (f * dirs[:, :, None, :]).sum(dim=-1)  # d . f  (E, B, C)
    ff = (f * f).sum(dim=-1)  # |f|^2 (E, B, C)
    disc = df * df - ff + radius * radius  # discriminant (E, B, C)
    hit = disc >= 0

    root = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = -df - root  # near intersection
    t2 = -df + root  # far intersection
    # Nearest positive intersection along the ray.
    t = torch.where(t1 > 0.0, t1, torch.where(t2 > 0.0, t2, torch.full_like(t1, float("inf"))))
    t = torch.where(hit, t, torch.full_like(t, float("inf")))
    t = torch.clamp(t, max=max_distance)
    return t.min(dim=-1).values  # (E, B)


def footprint_hits_cylinders(
    center_xy: torch.Tensor,
    heading: torch.Tensor,
    half_length: float,
    half_width: float,
    cylinder_centers: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Exact oriented-rectangle vs circle test.

    Args:
        center_xy: (E, 2) robot center [m].
        heading: (E,) robot heading [rad].
        half_length, half_width: robot footprint half-extents [m].
        cylinder_centers: (C, 2) cylinder centers [m].
        radius: cylinder radius [m].
    Returns:
        (E,) bool — True if any cylinder intersects the robot footprint.
    """
    c = torch.cos(heading)
    s = torch.sin(heading)
    # Vector from robot center to each cylinder, rotated into robot frame.
    rel = cylinder_centers[None, :, :] - center_xy[:, None, :]  # (E, C, 2)
    rx = c[:, None] * rel[..., 0] + s[:, None] * rel[..., 1]  # forward axis
    ry = -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]  # lateral axis
    # Closest point on the axis-aligned robot rectangle (in robot frame).
    cx = rx.clamp(-half_length, half_length)
    cy = ry.clamp(-half_width, half_width)
    dx = rx - cx
    dy = ry - cy
    dist2 = dx * dx + dy * dy  # (E, C)
    return (dist2 <= radius * radius).any(dim=-1)  # (E,)
