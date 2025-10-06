import mmcv
import mmengine
import mmseg

print("mmseg", getattr(mmseg, "__version__", "n/a"))
print("mmengine", mmengine.__version__)
print("mmcv", mmcv.__version__)


## Para modo CPU unicamente
from mmseg.apis import init_model

cfg = "./_mmseg_demo/pspnet_r50-d8_4xb4-80k_ade20k-512x512.py"
ckpt = "./_mmseg_demo/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth"
init_model(cfg, checkpoint=ckpt, device="cpu")
print("MMSeg model loaded OK.")
