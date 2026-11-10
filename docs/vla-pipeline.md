# VLA pipeline: demos → remote fine-tune → local inference

Isaac Lab already includes VLA-related tooling, but **no openpi / pi0** — that would need to be brought in separately.

## Stage 1: Collect demos locally

Run your Isaac Lab task and record demonstrations (state-based tasks run fine on the RTX 5050).

## Stage 2: Convert to LeRobot format

The conversion script is at:

```
IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py
```

It converts Isaac Lab HDF5 demonstrations into LeRobot (GR00t/GNx) format:
- parquet episodes
- H.264 videos
- `meta/info.json`, `meta/modality.json`, `meta/episodes.jsonl`

## Stage 3: Train remotely

Fine-tune a VLA (e.g. smol-vLA with LoRA/QLoRA) on a machine with **24+ GB VRAM** — Colab/Kaggle free tier, or a rented GPU (RunPod/vast.ai/Lambda).

RLinf VLA post-training docs describe fine-tuning GR00T and OpenVLA (PPO / Actor-Critic / SAC, FSDP) on Isaac Lab tasks:

```
IsaacLab/docs/source/experimental-features/rlinf_vla_posttraining.rst
```

## Stage 4: Bring checkpoint back

Download the fine-tuned checkpoint from Hugging Face Hub / Google Drive and run inference locally on the RTX 5050, or run a remote VLA inference server.

**Hardware constraint:** local training is not feasible on 8 GB VRAM; the laptop handles demo collection, LeRobot conversion, and inference only.
