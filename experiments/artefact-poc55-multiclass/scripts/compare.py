"""
POC-5.5 Model Comparison Script
Compares 3 hierarchical models (ConvNeXt, Swin, MaxViT) with multiclass metrics.

Usage:
    python compare.py [--logs logs] [--output logs/comparison]
"""

import os
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class HierarchicalModelComparison:
    """Compare hierarchical multiclass models"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.models = {}
        self.model_order = []
        
    def discover_models(self) -> List[str]:
        """Discover trained models from logs directory"""
        model_dirs = []
        for item in self.logs_dir.iterdir():
            if item.is_dir() and item.name != "comparison":
                if (item / "checkpoints" / "best_model.pth").exists():
                    model_dirs.append(item.name)
        
        print(f"📊 Discovered {len(model_dirs)} models: {', '.join(model_dirs)}")
        return sorted(model_dirs)
    
    def load_model_data(self, model_name: str) -> Dict:
        """Load all data for a model"""
        model_path = self.logs_dir / model_name
        
        data = {
            "name": model_name,
            "display_name": self._format_model_name(model_name),
        }
        
        # Load training log
        training_log_path = model_path / "logs" / "training_log.csv"
        if training_log_path.exists():
            data["training_log"] = pd.read_csv(training_log_path)
            print(f"  ✅ Training log: {len(data['training_log'])} epochs")
        else:
            print(f"  ❌ Missing training log")
            data["training_log"] = None
        
        # Load evaluation metrics
        metrics_path = model_path / "evaluation" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data["metrics"] = json.load(f)
            print(f"  ✅ Evaluation metrics loaded")
        else:
            print(f"  ❌ Missing evaluation metrics")
            data["metrics"] = None
        
        # Checkpoint info
        checkpoint_path = model_path / "checkpoints" / "best_model.pth"
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            data["checkpoint_size_mb"] = size_mb
            print(f"  ✅ Checkpoint: {size_mb:.1f} MB")
        else:
            data["checkpoint_size_mb"] = None
        
        return data
    
    def _format_model_name(self, model_name: str) -> str:
        """Format model name for display"""
        # Extract encoder name from experiment name
        if 'convnext' in model_name.lower():
            return "ConvNeXt-Tiny"
        elif 'swin' in model_name.lower():
            return "Swin-Tiny"
        elif 'maxvit' in model_name.lower():
            return "MaxViT-Tiny"
        else:
            return model_name.replace("_", " ").title()
    
    def load_all_models(self):
        """Discover and load all models"""
        model_names = self.discover_models()
        
        for model_name in model_names:
            print(f"\n📦 Loading {model_name}...")
            self.models[model_name] = self.load_model_data(model_name)
            self.model_order.append(model_name)
        
        print(f"\n✅ Loaded {len(self.models)} models\n")
    
    def create_metrics_table(self) -> pd.DataFrame:
        """Create comparison table from hierarchical evaluation metrics"""
        rows = []
        
        for model_name in self.model_order:
            model = self.models[model_name]
            metrics = model.get("metrics", {})
            
            if metrics:
                row = {
                    "Model": model["display_name"],
                    # Fine head (main task)
                    "mIoU Fine": metrics.get("fine", {}).get("miou", 0),
                    "mF1 Fine": metrics.get("fine", {}).get("mf1", 0),
                    # Coarse head
                    "mIoU Coarse": metrics.get("coarse", {}).get("miou", 0),
                    "mF1 Coarse": metrics.get("coarse", {}).get("mf1", 0),
                    # Binary head
                    "mIoU Binary": metrics.get("binary", {}).get("miou", 0),
                    "mF1 Binary": metrics.get("binary", {}).get("mf1", 0),
                    # Checkpoint
                    "Size (MB)": model.get("checkpoint_size_mb", 0),
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by fine mIoU (main metric)
        df = df.sort_values("mIoU Fine", ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_hierarchical_metrics_comparison(self, output_path: Path):
        """Plot hierarchical metrics comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = [self.models[m]["display_name"] for m in self.model_order]
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))
        
        # Prepare data for all 3 heads
        metrics_data = {
            "Binary mIoU": [],
            "Binary mF1": [],
            "Coarse mIoU": [],
            "Coarse mF1": [],
            "Fine mIoU": [],
            "Fine mF1": [],
        }
        
        for model_name in self.model_order:
            metrics = self.models[model_name].get("metrics", {})
            metrics_data["Binary mIoU"].append(metrics.get("binary", {}).get("miou", 0) * 100)
            metrics_data["Binary mF1"].append(metrics.get("binary", {}).get("mf1", 0) * 100)
            metrics_data["Coarse mIoU"].append(metrics.get("coarse", {}).get("miou", 0) * 100)
            metrics_data["Coarse mF1"].append(metrics.get("coarse", {}).get("mf1", 0) * 100)
            metrics_data["Fine mIoU"].append(metrics.get("fine", {}).get("miou", 0) * 100)
            metrics_data["Fine mF1"].append(metrics.get("fine", {}).get("mf1", 0) * 100)
        
        # Plot bars
        plots = [
            (metrics_data["Binary mIoU"], "Binary mIoU (%)", axes[0, 0]),
            (metrics_data["Binary mF1"], "Binary mF1 (%)", axes[0, 1]),
            (metrics_data["Coarse mIoU"], "Coarse mIoU (%)", axes[0, 2]),
            (metrics_data["Coarse mF1"], "Coarse mF1 (%)", axes[1, 0]),
            (metrics_data["Fine mIoU"], "Fine mIoU (%)", axes[1, 1]),
            (metrics_data["Fine mF1"], "Fine mF1 (%)", axes[1, 2]),
        ]
        
        for values, title, ax in plots:
            bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, max(values) * 1.15] if values else [0, 100])
        
        plt.suptitle("POC-5.5 Hierarchical Metrics Comparison", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved hierarchical metrics: {output_path}")
    
    def plot_training_curves(self, output_path: Path):
        """Plot training curves for all 3 heads"""
        if not any(self.models[m].get("training_log") is not None for m in self.model_order):
            print("⚠️  No training logs found, skipping training curves")
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))
        
        metrics = [
            ("epoch", "train_loss", "Training Loss", axes[0, 0]),
            ("epoch", "val_loss", "Validation Loss", axes[0, 1]),
            ("epoch", "val_miou_binary", "Val mIoU Binary", axes[0, 2]),
            ("epoch", "val_miou_coarse", "Val mIoU Coarse", axes[0, 3]),
            ("epoch", "val_miou_fine", "Val mIoU Fine", axes[1, 0]),
            ("epoch", "val_mdice_binary", "Val mDice Binary", axes[1, 1]),
            ("epoch", "val_mdice_coarse", "Val mDice Coarse", axes[1, 2]),
            ("epoch", "val_mdice_fine", "Val mDice Fine", axes[1, 3]),
        ]
        
        for i, model_name in enumerate(self.model_order):
            model = self.models[model_name]
            df = model.get("training_log")
            
            if df is None:
                continue
            
            color = colors[i]
            label = model["display_name"]
            
            for x_col, y_col, title, ax in metrics:
                if x_col in df.columns and y_col in df.columns:
                    ax.plot(df[x_col], df[y_col], 
                           label=label, color=color, linewidth=2, alpha=0.8)
        
        # Configure axes
        for x_col, y_col, title, ax in metrics:
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("POC-5.5 Training Curves", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved training curves: {output_path}")
    
    def export_comparison_table(self, output_path: Path):
        """Export comparison table"""
        df = self.create_metrics_table()
        
        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved CSV: {csv_path}")
        
        # Save formatted text
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("POC-5.5 HIERARCHICAL MODEL COMPARISON - SUMMARY TABLE\n")
            f.write("=" * 120 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Add rankings
            f.write("=" * 120 + "\n")
            f.write("RANKINGS (by Fine mIoU - Main Task)\n")
            f.write("=" * 120 + "\n\n")
            
            for idx, row in df.iterrows():
                medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"{idx+1}."
                f.write(f"{medal} {row['Model']}: {row['mIoU Fine']:.4f} mIoU (Fine)\n")
                f.write(f"    Coarse: {row['mIoU Coarse']:.4f}, Binary: {row['mIoU Binary']:.4f}\n\n")
        
        print(f"✅ Saved TXT: {txt_path}")
        
        return df
    
    def create_summary_report(self, output_path: Path):
        """Create comprehensive summary report"""
        with open(output_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("POC-5.5: HIERARCHICAL MULTICLASS SEGMENTATION - COMPREHENSIVE REPORT\n")
            f.write("=" * 120 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Compared: {len(self.models)}\n")
            f.write(f"Innovation: Hierarchical Multi-Task Learning (3 heads: Binary/Coarse/Fine)\n\n")
            
            # Overall winner
            df = self.create_metrics_table()
            if len(df) > 0:
                winner = df.iloc[0]
                
                f.write("=" * 120 + "\n")
                f.write("🏆 WINNER (Best Fine mIoU)\n")
                f.write("=" * 120 + "\n\n")
                f.write(f"Model: {winner['Model']}\n")
                f.write(f"Fine Head:   mIoU={winner['mIoU Fine']:.4f} ({winner['mIoU Fine']*100:.2f}%), mF1={winner['mF1 Fine']:.4f}\n")
                f.write(f"Coarse Head: mIoU={winner['mIoU Coarse']:.4f} ({winner['mIoU Coarse']*100:.2f}%), mF1={winner['mF1 Coarse']:.4f}\n")
                f.write(f"Binary Head: mIoU={winner['mIoU Binary']:.4f} ({winner['mIoU Binary']*100:.2f}%), mF1={winner['mF1 Binary']:.4f}\n\n")
            
            # Detailed results per model
            f.write("=" * 120 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 120 + "\n\n")
            
            for model_name in self.model_order:
                model = self.models[model_name]
                metrics = model.get("metrics", {})
                
                f.write(f"📊 {model['display_name']}\n")
                f.write("-" * 120 + "\n")
                
                if metrics:
                    # Binary head
                    f.write("Binary Head (Clean vs Damage):\n")
                    binary = metrics.get('binary', {})
                    f.write(f"  mIoU: {binary.get('miou', 0):.4f}, mF1: {binary.get('mf1', 0):.4f}\n")
                    iou_binary = binary.get('iou_per_class', [])
                    if len(iou_binary) >= 2:
                        f.write(f"  Clean IoU: {iou_binary[0]:.4f}, Damage IoU: {iou_binary[1]:.4f}\n\n")
                    
                    # Coarse head
                    f.write("Coarse Head (4 Damage Groups):\n")
                    coarse = metrics.get('coarse', {})
                    f.write(f"  mIoU: {coarse.get('miou', 0):.4f}, mF1: {coarse.get('mf1', 0):.4f}\n")
                    iou_coarse = coarse.get('iou_per_class', [])
                    coarse_names = ['Structural', 'Surface', 'Color', 'Optical']
                    for i, name in enumerate(coarse_names):
                        if i < len(iou_coarse):
                            f.write(f"  {name}: {iou_coarse[i]:.4f}\n")
                    f.write("\n")
                    
                    # Fine head
                    f.write("Fine Head (16 Classes):\n")
                    fine = metrics.get('fine', {})
                    f.write(f"  mIoU: {fine.get('miou', 0):.4f}, mF1: {fine.get('mf1', 0):.4f}\n")
                    iou_fine = fine.get('iou_per_class', [])
                    class_names = ['Clean', 'Material loss', 'Peel', 'Cracks', 'Structural defects',
                                  'Dirt spots', 'Stains', 'Discolouration', 'Scratches', 'Burn marks',
                                  'Hairs', 'Dust spots', 'Lightleak', 'Fading', 'Blur', 'Other damage']
                    for i, name in enumerate(class_names):
                        if i < len(iou_fine):
                            f.write(f"  {i:2d}. {name:20s}: {iou_fine[i]:.4f}\n")
                
                f.write(f"\nCheckpoint Size: {model.get('checkpoint_size_mb', 0):.1f} MB\n\n")
        
        print(f"✅ Saved comprehensive report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare POC-5.5 hierarchical models")
    parser.add_argument("--logs", default="logs", help="Logs directory")
    parser.add_argument("--output", default="logs/comparison", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Initialize comparison
    comparison = HierarchicalModelComparison(logs_dir=args.logs)
    
    # Load all models
    comparison.load_all_models()
    
    if len(comparison.models) == 0:
        print("❌ No models found! Train models first.")
        return 1
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    comparison.plot_hierarchical_metrics_comparison(output_dir / "hierarchical_metrics.png")
    comparison.plot_training_curves(output_dir / "training_curves.png")
    
    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80 + "\n")
    
    comparison.export_comparison_table(output_dir / "comparison_table")
    comparison.create_summary_report(output_dir / "summary_report.txt")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ POC-5.5 COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - hierarchical_metrics.png")
    print("  - training_curves.png")
    print("  - comparison_table.csv")
    print("  - comparison_table.txt")
    print("  - summary_report.txt")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
