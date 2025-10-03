from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import torch

# Transformer
config_file = './checkpoints/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py'
checkpoint_file = './checkpoints/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device)
# device='cuda:0' only for machines with GPU

# test a single image and show the results
img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)

# --------------------------------------------
# CNN
config_file = './checkpoints/convnext-large_upernet_8xb2-amp-160k_ade20k-640x640.py'
checkpoint_file = '.checkpoints/upernet_convnext_large_fp16_640x640_160k_ade20k_20220226_040532-e57aa54d.pth'

model = init_model(config_file, checkpoint_file, device)
result = inference_model(model, img)
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
