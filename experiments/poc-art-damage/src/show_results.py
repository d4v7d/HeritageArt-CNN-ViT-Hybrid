#!/usr/bin/env python3
"""Display summary of pipeline results in a nice format."""

import json
from pathlib import Path
import sys


def load_results(results_dir: Path):
    """Load all results from pipeline."""
    summary_file = results_dir / 'summary_results.json'
    
    if not summary_file.exists():
        print(f"Error: {summary_file} not found!")
        print("Make sure the pipeline has completed successfully.")
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def print_comparison_table(results):
    """Print nice comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<25} {'mIoU':<20} {'mF1':<20} {'# Images':<10}")
    print("-" * 80)
    
    # Data rows
    for result in results:
        model = result['model_name']
        miou = f"{result['mean_mIoU']:.4f} ± {result['std_mIoU']:.4f}"
        mf1 = f"{result['mean_mF1']:.4f} ± {result['std_mF1']:.4f}"
        n_imgs = result['num_images']
        
        print(f"{model:<25} {miou:<20} {mf1:<20} {n_imgs:<10}")
    
    print("-" * 80)
    
    # Find best models
    best_miou = max(results, key=lambda x: x['mean_mIoU'])
    best_mf1 = max(results, key=lambda x: x['mean_mF1'])
    
    print(f"\n🏆 Best mIoU: {best_miou['model_name']} ({best_miou['mean_mIoU']:.4f})")
    print(f"🏆 Best mF1:  {best_mf1['model_name']} ({best_mf1['mean_mF1']:.4f})")


def print_per_image_details(results_dir: Path, model_name: str, top_n: int = 5):
    """Print details for specific model."""
    model_dir = results_dir / model_name
    overall_file = model_dir / 'overall_metrics.json'
    
    if not overall_file.exists():
        print(f"\nWarning: {overall_file} not found")
        return
    
    with open(overall_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n\n{'='*80}")
    print(f"DETAILED RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"\nOverall Statistics:")
    print(f"  Mean mIoU: {data['mean_mIoU']:.4f} ± {data['std_mIoU']:.4f}")
    print(f"  Mean mF1:  {data['mean_mF1']:.4f} ± {data['std_mF1']:.4f}")
    print(f"  Total images: {data['num_images']}")
    
    # Sort images by mIoU
    images = data['per_image_metrics']
    images_sorted = sorted(images, key=lambda x: x['mIoU'], reverse=True)
    
    print(f"\n📊 Top {top_n} images by mIoU:")
    print(f"{'Image ID':<15} {'mIoU':<12} {'mF1':<12}")
    print("-" * 40)
    for img in images_sorted[:top_n]:
        print(f"{img['image_id']:<15} {img['mIoU']:<12.4f} {img['mF1']:<12.4f}")
    
    print(f"\n📉 Bottom {top_n} images by mIoU:")
    print(f"{'Image ID':<15} {'mIoU':<12} {'mF1':<12}")
    print("-" * 40)
    for img in images_sorted[-top_n:]:
        print(f"{img['image_id']:<15} {img['mIoU']:<12.4f} {img['mF1']:<12.4f}")


def print_file_structure(results_dir: Path):
    """Print information about generated files."""
    print(f"\n\n{'='*80}")
    print("GENERATED FILES")
    print(f"{'='*80}")
    
    print(f"\n📁 Results directory: {results_dir}")
    
    checkpoints = results_dir / 'checkpoints'
    if checkpoints.exists():
        print(f"\n🔖 Checkpoints:")
        for ckpt in sorted(checkpoints.glob('*.pth')):
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
    
    print(f"\n📸 Visualizations by model:")
    for model_dir in sorted(results_dir.glob('*')):
        if model_dir.is_dir() and model_dir.name != 'checkpoints':
            vis_files = list(model_dir.glob('*_visualization.png'))
            metric_files = list(model_dir.glob('*_metrics.json'))
            print(f"  - {model_dir.name}:")
            print(f"      {len(vis_files)} visualization images")
            print(f"      {len(metric_files)} metric files")


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path('logs/pipeline_results')
    
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found!")
        print("Usage: python -m src.show_results [results_dir]")
        return
    
    # Load and display results
    results = load_results(results_dir)
    if results is None:
        return
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print details for each model
    for result in results:
        print_per_image_details(results_dir, result['model_name'], top_n=3)
    
    # Print file structure
    print_file_structure(results_dir)
    
    print(f"\n\n✅ Complete results available at: {results_dir}")
    print("Check the visualization PNGs for detailed segmentation results!")


if __name__ == '__main__':
    main()
