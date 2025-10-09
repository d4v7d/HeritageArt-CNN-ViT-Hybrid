import mmcv
import mmengine
import mmseg

import os
from pathlib import Path

print("Versions Loaded:")
print("mmseg", getattr(mmseg, "__version__", "n/a"))
print("mmengine", mmengine.__version__)
print("mmcv", mmcv.__version__)
print("\n")

# Para modo CPU unicamente
from mmseg.apis import init_model

# Resolve paths relative to this script, not terminal
HERE = Path(__file__).resolve().parent
# .../experiments/mmseg_demos/
ROOT = HERE.parents[1]
CKPTS = ROOT / "checkpoints"
EXPTS = ROOT / "experiments"

def must_exist(p: Path):
  if not p.exists():
    raise FileNotFoundError(f"Missing: {p}")
  return p

print("Test básico de carga")
print("cwd:", Path.cwd())
print("script dir:", HERE)
print("root:", ROOT)
print("experiments:", EXPTS)
print("\n")

# experiments/mmseg_demos/_mmseg_demo/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py
cfg = must_exist(EXPTS / "mmseg_demos/_mmseg_demo/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py")
# experiments/mmseg_demos/_mmseg_demo/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth
ckpt = must_exist(EXPTS / "mmseg_demos/_mmseg_demo/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth")

init_model(str(cfg), checkpoint=str(ckpt), device="cpu")
print("\nMMSeg model loaded OK.")