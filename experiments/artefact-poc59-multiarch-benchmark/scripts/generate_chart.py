
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from the JSON logs
classes = [
    'Clean', 'Material_loss', 'Peel', 'Dust', 'Scratch',
    'Hair', 'Dirt', 'Fold', 'Writing', 'Cracks',
    'Staining', 'Stamp', 'Sticker', 'Puncture', 'Burn_marks', 'Lightleak'
]

# IoU values extracted from logs
cnn_iou = [
    0.928, 0.707, 0.520, 0.0, 0.026, 
    0.0, 0.079, 0.328, 0.249, 0.102, 
    0.138, 0.0, 0.625, 0.690, 0.0, 0.518
]

vit_iou = [
    0.945, 0.777, 0.611, 0.0, 0.205, 
    0.0, 0.149, 0.489, 0.506, 0.285, 
    0.419, 0.0, 0.685, 0.0, 0.00006, 0.632
]

hybrid_iou = [
    0.950, 0.819, 0.653, 0.0, 0.236, 
    0.0, 0.257, 0.504, 0.532, 0.290, 
    0.525, 0.606, 0.0, 0.0, 0.0, 0.637
]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width, cnn_iou, width, label='CNN (ConvNeXt)', color='#1f77b4')
rects2 = ax.bar(x, vit_iou, width, label='ViT (SegFormer)', color='#ff7f0e')
rects3 = ax.bar(x + width, hybrid_iou, width, label='Hybrid (MaxViT)', color='#2ca02c')

ax.set_ylabel('IoU Score')
ax.set_title('Per-Class IoU Comparison by Architecture')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(0, 1.0)

# Add some text for labels, title and custom x-axis tick labels, etc.
fig.tight_layout()

output_path = '/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/documentation/figures/per_class_iou_chart.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Chart saved to {output_path}")
