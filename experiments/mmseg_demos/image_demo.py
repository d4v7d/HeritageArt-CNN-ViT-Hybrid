import mmcv
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot

# Transformer
config_file = "../../checkpoints/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py"
checkpoint_file = "../../checkpoints/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device)

# test a single image
img = "../../demo/demo.png"  # Asume que demo.png está en la carpeta demo/ original
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True, out_file="result.jpg", opacity=0.5)

print("\nFinalized SWIN\nStarting ConvNeXt\n")

# --------------------------------------------
# CNN
config_file = "../../checkpoints/convnext-large_upernet_8xb2-amp-160k_ade20k-640x640.py"
checkpoint_file = "../../checkpoints/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth"

model = init_model(config_file, checkpoint_file, device)
result = inference_model(model, img)
show_result_pyplot(model, img, result, show=True, out_file="result.jpg", opacity=0.5)