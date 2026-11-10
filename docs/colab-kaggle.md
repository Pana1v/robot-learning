# Free cloud GPU: Colab & Kaggle

Your RTX 5050 Laptop (8 GB VRAM) cannot train VLA models locally. Use free cloud GPU for the training step, then bring the checkpoint back for local inference.

## Google Colab (free)

- Free T4 GPU (~16 GB VRAM), sessions recycle after ~12 hours (often less) and disconnect after ~90 minutes idle.
- Good for short LoRA/QLoRA fine-tuning runs of small VLAs (e.g. smol-vLA).
- Not guaranteed for multi-day training.

**Tips to stretch a free session:**
- Don't close the browser tab; avoid letting the computer sleep.
- Interact occasionally (click/scroll a cell) so Colab registers activity.
- **Checkpoint aggressively** to Google Drive or Hugging Face Hub every N steps — an interruption must not lose progress.
- Keep notebook output lightweight.

```python
from google.colab import drive
drive.mount('/content/drive')
# save checkpoints to /content/drive/MyDrive/...
```

**Do not** use clickers/heartbeat scripts to bypass idle disconnects — that violates Colab's terms of service.

## Kaggle Kernels (free)

- ~30 GPU hours/week (T4/x2), weekly cap resets.
- Notebook-based fine-tuning; better for longer runs than Colab free.
- Dataset/model size limits apply.

## Hugging Face Hub (free)

- Host your fine-tuned checkpoint and dataset; download back to the laptop.
- Use it as the checkpoint staging area between remote training and local inference.

## build.nvidia.com API

Rate-limited to ~40 requests/minute (~1 request / 1.5 s). Too slow for real-time control loops (30–120 Hz). Use only for offline inference, data generation, or occasional planning. For real-time robot learning, run a smaller quantized model locally.

## Recommended split

| Step | Where | Cost |
|------|-------|------|
| Collect Isaac Lab demos | local laptop | free |
| Convert to LeRobot format | local laptop | free |
| Upload dataset | Hugging Face Hub | free |
| Fine-tune VLA (QLoRA/LoRA) | Colab / Kaggle / rented GPU | free / $0.30–0.70/hr |
| Download checkpoint | local laptop | free |
| Run inference | RTX 5050 locally | free |

## Alternatives if you need sustained multi-day training

- Colab Pro / Pro+ (paid): longer runs, priority GPU.
- RunPod / vast.ai / Lambda Cloud: rented GPU by the hour.
- Lightning.ai: free studio hours with GPU option.
