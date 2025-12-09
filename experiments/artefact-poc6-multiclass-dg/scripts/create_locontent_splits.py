import pandas as pd
import json
from pathlib import Path
import os
import re

def create_splits():
    # Paths
    # Assuming running from experiments/artefact-poc6-multiclass-dg
    metadata_path = Path('../common-data/artefact_original/metadata.csv')
    augmented_images_dir = Path('../common-data/artefact_augmented/images')
    
    if not metadata_path.exists():
        # Try absolute path
        metadata_path = Path('/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/common-data/artefact_original/metadata.csv')
        augmented_images_dir = Path('/opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/common-data/artefact_augmented/images')
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        return

    print(f"Reading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Create ID -> Content map
    # ID is the filename without extension in the original dataset
    # metadata.csv has 'id' column
    id_to_content = dict(zip(df['id'], df['content']))
    
    print(f"Loaded {len(id_to_content)} original IDs")
    
    # Scan augmented images
    print(f"Scanning images in {augmented_images_dir}")
    image_files = sorted([f.name for f in augmented_images_dir.glob('*.png')])
    print(f"Found {len(image_files)} augmented images")
    
    # Map images to content
    image_content_map = []
    
    for img_file in image_files:
        # Extract ID
        # Format: id.png or id_augX.png
        # Regex to match ID
        # Assuming ID is alphanumeric
        # If id contains underscores, this might be tricky.
        # But looking at 'cljmrkz5n341f07clcujw105j', it seems to be a hash.
        
        # Split by '_aug' if present, else remove extension
        if '_aug' in img_file:
            img_id = img_file.split('_aug')[0]
        else:
            img_id = os.path.splitext(img_file)[0]
            
        content = id_to_content.get(img_id)
        
        if content:
            image_content_map.append({
                'image_path': f"images/{img_file}", # Relative to data root
                'content': content
            })
        else:
            print(f"⚠️  Warning: ID {img_id} not found in metadata (file: {img_file})")
            
    df_aug = pd.DataFrame(image_content_map)
    
    # Get unique content types
    content_types = sorted(df_aug['content'].unique())
    print(f"Found content types: {content_types}")
    
    manifests_dir = Path('manifests')
    manifests_dir.mkdir(exist_ok=True)
    
    for i, held_out_content in enumerate(content_types):
        fold_id = i + 1
        print(f"Creating Fold {fold_id}: Held-out = {held_out_content}")
        
        # Split
        test_df = df_aug[df_aug['content'] == held_out_content]
        train_df = df_aug[df_aug['content'] != held_out_content]
        
        # Create manifest
        manifest = {
            'fold': fold_id,
            'held_out_content': held_out_content,
            'train': train_df['image_path'].tolist(),
            'test': test_df['image_path'].tolist(),
            'train_count': len(train_df),
            'test_count': len(test_df)
        }
        
        # Save
        output_path = manifests_dir / f'locontent_fold{fold_id}.json'
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"  Saved to {output_path}")
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

if __name__ == '__main__':
    create_splits()
