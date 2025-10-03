from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = './checkpoints/swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512.py'
checkpoint_file = './checkpoints/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_model(model, frame)
   show_result_pyplot(model, frame, result, wait_time=1)