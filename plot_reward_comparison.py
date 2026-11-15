"""
Compare OLD vs NEW reward function training runs.

Produces:
  1. Training curve: mean RL cost vs training steps (from eval logs), old vs new.
  2. Final head-to-head: load both models, evaluate on the SAME eval scenarios,
     plot mean RL cost vs greedy baseline for old vs new.
  3. Saves figure to leap_reward_comparison.png and a JSON summary.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_DIR = Path("/home/panav/isaaclab-barn")
OLD_LOG = RUN_DIR / "leap_run_old.log"
NEW_LOG = RUN_DIR / "leap_run_new.log"
OLD_MODEL = RUN_DIR / "leap_run_old.zip"
NEW_MODEL = RUN_DIR / "leap_run_new.zip"
OLD_REPORT = RUN_DIR / "leap_run_old_report.json"
NEW_REPORT = RUN_DIR / "leap_run_new_report.json"
OUT_PNG = RUN_DIR / "leap_reward_comparison.png"
OUT_JSON = RUN_DIR / "leap_reward_comparison_summary.json"

# ---------------------------------------------------------------------------
# Parse eval lines from training logs.
# Format:  [eval #1 | 40,000 steps]  RL=338.3  Greedy=330.8  Δ=-2.26% ↑
# ---------------------------------------------------------------------------
_EVAL_RE = re.compile(
    r"\[eval #(?P<num>\d+) \| (?P<step>[\d,]+) steps\]\s+RL=(?P<rl>[\d.]+)\s+Greedy=(?P<gr>[\d.]+)"
)


def parse_eval_curve(log_path: Path):
    steps, rl_costs, greedy_costs, deltas = [], [], [], []
    if not log_path.exists():
        return steps, rl_costs, greedy_costs, deltas
    with open(log_path) as f:
        for line in f:
            m = _EVAL_RE.search(line)
            if not m:
                continue
            steps.append(int(m.group("step").replace(",", "")))
            rl_costs.append(float(m.group("rl")))
            greedy_costs.append(float(m.group("gr")))
            deltas.append((float(m.group("gr")) - float(m.group("rl"))) / float(m.group("gr")) * 100.0)
    return steps, rl_costs, greedy_costs, deltas


# ---------------------------------------------------------------------------
# Final head-to-head evaluation: load both models, same scenarios.
# ---------------------------------------------------------------------------
def head_to_head_eval():
    """Load both models and evaluate on the same scenarios; return summary dict."""
    try:
        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from binning_env import BinningEnv, CostMinBinningEnv  # noqa: F401  (env registered via import)
    except Exception as exc:
        print(f"[plot] cannot import RL deps for head-to-head eval: {exc}")
        return None

    sys_path = Path(__file__).resolve().parent
    import sys
    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    # Ensure src/ (binning_env) is importable
    src = Path("/home/panav/Documents/repositories/research-papers/LEAP/src")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from binning_env import BinningEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build a shared eval scenario set (3 objects, 200 scenarios, seed 0 to match training)
    rng = np.random.default_rng(0)
    eval_scenarios = []
    for _ in range(200):
        objects = rng.uniform(10.0, 90.0, size=(3, 2)).astype(np.float32)
        types = rng.integers(0, 4, size=3).astype(np.int32)
        bins = np.array([[10, 10], [90, 10], [10, 90], [90, 90]], dtype=np.float32)
        start = rng.uniform(10.0, 90.0, size=2).astype(np.float32)
        # greedy cost via nearest-neighbour
        import math
        remaining = set(range(3))
        pos = start.copy()
        cost = 0.0
        while remaining:
            best_i, best_v = -1, math.inf
            for i in remaining:
                d = np.linalg.norm(pos - objects[i]) + np.linalg.norm(objects[i] - bins[types[i]])
                if d < best_v:
                    best_v, best_i = d, i
            obj = objects[best_i]
            bpos = bins[types[best_i]]
            cost += float(np.linalg.norm(pos - obj) + np.linalg.norm(obj - bpos))
            pos = bpos.copy()
            remaining.discard(best_i)
        eval_scenarios.append({
            "objects": objects.tolist(),
            "types": types.tolist(),
            "bins": bins.tolist(),
            "start": start.tolist(),
            "greedy_cost": cost,
        })

    results = {}
    for label, model_path in (("old", OLD_MODEL), ("new", NEW_MODEL)):
        if not model_path.exists():
            print(f"[plot] {label} model not found: {model_path}")
            continue
        model = MaskablePPO.load(str(model_path), device=device)
        costs = []
        for sc in eval_scenarios:
            env = BinningEnv(sc, max_objects=3)
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 12:
                act, _ = model.predict(obs, deterministic=True, action_masks=env.get_action_mask())
                obs, _, term, trunc, info = env.step(int(act))
                done = term or trunc
                steps += 1
            invalid = info.get("invalid_action", False)
            ok = (not trunc) and (not invalid) and (len(env.remaining_objects) == 0)
            costs.append(env.total_distance if ok else sc["greedy_cost"] * 2.0)
        costs = np.array(costs)
        greedy = np.array([s["greedy_cost"] for s in eval_scenarios])
        results[label] = {
            "rl_mean": float(costs.mean()),
            "rl_std": float(costs.std()),
            "greedy_mean": float(greedy.mean()),
            "delta_pct": float((greedy.mean() - costs.mean()) / greedy.mean() * 100.0),
            "beats_greedy_pct": float(np.mean(costs < greedy) * 100.0),
        }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main():
    old_steps, old_rl, old_gr, old_delta = parse_eval_curve(OLD_LOG)
    new_steps, new_rl, new_gr, new_delta = parse_eval_curve(NEW_LOG)

    h2h = head_to_head_eval()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: training curve RL cost vs steps
    ax = axes[0]
    if old_steps:
        ax.plot(old_steps, old_rl, marker="o", label="OLD reward", color="#1f77b4")
    if new_steps:
        ax.plot(new_steps, new_rl, marker="s", label="NEW reward", color="#d62728")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean RL cost (eval)")
    ax.set_title("RL Cost vs Training Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: delta vs greedy over training
    ax = axes[1]
    if old_steps:
        ax.plot(old_steps, old_delta, marker="o", label="OLD reward", color="#1f77b4")
    if new_steps:
        ax.plot(new_steps, new_delta, marker="s", label="NEW reward", color="#d62728")
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Delta vs greedy (%)")
    ax.set_title("Improvement vs Greedy During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: final head-to-head bar chart
    ax = axes[2]
    labels, rl_means, greedy_means = [], [], []
    if h2h:
        for label in ("old", "new"):
            if label in h2h:
                labels.append(label.upper())
                rl_means.append(h2h[label]["rl_mean"])
                greedy_means.append(h2h[label]["greedy_mean"])
    if labels:
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, rl_means, w, label="RL mean cost", color=["#1f77b4", "#d62728"])
        ax.bar(x + w / 2, greedy_means, w, label="Greedy baseline", color="#888888", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean cost")
        ax.set_title("Final Head-to-Head (same 200 scenarios)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Head-to-head eval\nrequires both models", ha="center", va="center")
        ax.set_title("Final Head-to-Head")

    fig.suptitle("OLD vs NEW Reward Function — 3 objects, 1e6 timesteps", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[plot] saved figure → {OUT_PNG}")

    # Save JSON summary
    summary = {
        "old_eval_curve": {"steps": old_steps, "rl_mean_cost": old_rl, "greedy_mean_cost": old_gr, "delta_pct": old_delta},
        "new_eval_curve": {"steps": new_steps, "rl_mean_cost": new_rl, "greedy_mean_cost": new_gr, "delta_pct": new_delta},
        "head_to_head": h2h,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[plot] saved summary → {OUT_JSON}")


if __name__ == "__main__":
    main()
