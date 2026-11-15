#!/usr/bin/env python3
"""Build the Jackal visual-only USD asset from the Isaac Sim Clearpath Jackal.

The upstream Isaac asset (Clearpath Jackal, 4.5) is a full articulation: an
articulation root, five rigid bodies (chassis + four wheels) and joints. The
BARN env models the robot as ONE velocity-driven rigid body (Phase A, D2), so
the articulation cannot be spawned directly as the robot body.

This script downloads the upstream asset and strips every physics schema from
it, leaving the visual hierarchy only (meshes, materials, transforms). The
result is parented under the robot's rigid body as a visual-only child, so the
renderer draws the real Jackal while the physics stays the proven single-body
model.

Usage:
    python scripts/build_jackal_asset.py
    python scripts/build_jackal_asset.py --force
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Isaac Sim assets: Clearpath Jackal, shipped with the 4.5 asset drop.
JACKAL_URL = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/Robots/Clearpath/Jackal/jackal.usd"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / ".context" / "assets" / "jackal.usd"
OUT_PATH = REPO_ROOT / "source" / "my_robot_tasks" / "my_robot_tasks" / "assets" / "jackal_visual.usd"

# Physics APIs removed from the upstream asset (applied schemas).
PHYSICS_APIS = (
    UsdPhysics.RigidBodyAPI,
    UsdPhysics.ArticulationRootAPI,
    UsdPhysics.MassAPI,
    UsdPhysics.CollisionAPI,
    UsdPhysics.MaterialAPI,
)


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Fetch the upstream USD into the local cache (skip if already there)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[jackal-asset] cached upstream asset: {dest} ({dest.stat().st_size} bytes)")
        return dest
    print(f"[jackal-asset] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    print(f"[jackal-asset] downloaded {dest.stat().st_size} bytes -> {dest}")
    return dest


def _remove_property(prim, attr_name: str) -> None:
    """Remove an attribute spec from a prim (all layers that author it)."""
    prim.RemoveProperty(attr_name)


def strip_physics(stage: Usd.Stage) -> dict[str, int]:
    """Remove every physics schema and physics attribute from the stage.

    Returns a small report of what was removed.
    """
    report = {"apis": 0, "joints": 0, "attributes": 0, "environment": 0}

    # Drop the asset's own ground plane: the env owns the ground, and the
    # referenced default prim must be the only root that comes along.
    env_prim = stage.GetPrimAtPath("/Environment")
    if env_prim.IsValid():
        stage.RemovePrim("/Environment")
        report["environment"] = 1

    # Remove joint prims (typed schemas, not APIs) along with their subtrees.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            stage.RemovePrim(prim.GetPath())
            report["joints"] += 1

    for prim in list(stage.Traverse()):
        for api in PHYSICS_APIS:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                report["apis"] += 1
        # Drop leftover physics:* attributes (e.g. physics:rigidBodyEnabled).
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith("physics:"):
                _remove_property(prim, attr.GetName())
                report["attributes"] += 1

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=JACKAL_URL, help="Upstream Jackal USD URL.")
    parser.add_argument("--out", default=str(OUT_PATH), help="Output USD path.")
    parser.add_argument("--force", action="store_true", help="Re-download the upstream asset.")
    args = parser.parse_args()

    src = download(args.url, CACHE_PATH, force=args.force)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(src))
    if stage is None:
        raise RuntimeError(f"could not open upstream USD: {src}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("upstream Jackal USD has no default prim")
    print(f"[jackal-asset] upstream default prim: {default_prim.GetPath()}")

    report = strip_physics(stage)
    print(f"[jackal-asset] stripped: {report}")

    # Re-assert the default prim so the spawner imports /jackal only.
    stage.SetDefaultPrim(default_prim)

    stage.GetRootLayer().Export(str(out))

    # Verify the exported asset.
    check = Usd.Stage.Open(str(out))
    leftover = [
        str(p.GetPath())
        for p in check.Traverse()
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
        or p.HasAPI(UsdPhysics.ArticulationRootAPI)
        or p.HasAPI(UsdPhysics.MassAPI)
        or p.HasAPI(UsdPhysics.CollisionAPI)
        or p.IsA(UsdPhysics.Joint)
        or p.GetTypeName().startswith("Physics")
    ]
    if leftover:
        raise RuntimeError(f"physics schemas survived the strip: {leftover}")

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeLocalBound(
        check.GetDefaultPrim()
    ).ComputeAlignedRange()
    print(f"[jackal-asset] wrote {out} ({out.stat().st_size} bytes)")
    print(f"[jackal-asset] prims: {len(list(check.Traverse()))}")
    print(
        "[jackal-asset] local bbox min "
        f"{[round(v, 4) for v in bbox.GetMin()]} max {[round(v, 4) for v in bbox.GetMax()]}"
    )
    print(f"[jackal-asset] wheel contact plane z = {bbox.GetMin()[2]:.4f} (asset frame)")


if __name__ == "__main__":
    main()
