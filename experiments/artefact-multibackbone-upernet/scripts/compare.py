"""
POC-5 Model Comparison Script
Dynamically loads training logs and evaluation metrics to create comprehensive comparisons.

Usage:
    python compare.py [--output logs/comparison]
"""

import os
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Configure plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ModelComparison:
    """Load and compare multiple models dynamically"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.models = {}
        self.model_order = []  # Preserve order for consistent plotting
        
    def discover_models(self) -> List[str]:
        """Automatically discover trained models from logs directory"""
        model_dirs = []
        for item in self.logs_dir.iterdir():
            if item.is_dir() and item.name != "comparison":
                # Check if it has the required structure
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
        training_log_path = model_path / "training" / "training_log.csv"
        if training_log_path.exists():
            data["training_log"] = pd.read_csv(training_log_path)
            print(f"  ✅ Loaded training log: {len(data['training_log'])} epochs")
        else:
            print(f"  ❌ Missing training log")
            data["training_log"] = None
        
        # Load evaluation metrics
        metrics_path = model_path / "evaluation" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data["metrics"] = json.load(f)
            print(f"  ✅ Loaded evaluation metrics")
        else:
            print(f"  ❌ Missing evaluation metrics")
            data["metrics"] = None
        
        # Find best checkpoint info
        checkpoint_path = model_path / "checkpoints" / "best_model.pth"
        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            data["checkpoint_size_mb"] = size_mb
            
            # Find best epoch from training log
            if data["training_log"] is not None:
                best_idx = data["training_log"]["val_miou"].idxmax()
                data["best_epoch"] = int(data["training_log"].loc[best_idx, "epoch"])
                data["best_val_miou"] = float(data["training_log"].loc[best_idx, "val_miou"])
            print(f"  ✅ Checkpoint: {size_mb:.1f} MB, Best epoch: {data.get('best_epoch', '?')}")
        else:
            print(f"  ❌ Missing checkpoint")
            data["checkpoint_size_mb"] = None
        
        # Check for predictions
        predictions_path = model_path / "evaluation" / "predictions"
        if predictions_path.exists():
            pred_files = list(predictions_path.glob("*.png"))
            data["num_predictions"] = len(pred_files)
            data["prediction_files"] = pred_files
            print(f"  ✅ Found {len(pred_files)} prediction images")
        else:
            data["num_predictions"] = 0
            data["prediction_files"] = []
        
        return data
    
    def _format_model_name(self, model_name: str) -> str:
        """Convert model_name to display format"""
        # convnext_tiny_upernet -> ConvNeXt-Tiny
        # swin_tiny_upernet -> Swin-Tiny
        # maxvit_tiny_upernet -> MaxViT-Tiny
        
        name_map = {
            "convnext_tiny_upernet": "ConvNeXt-Tiny",
            "swin_tiny_upernet": "Swin-Tiny",
            "maxvit_tiny_upernet": "MaxViT-Tiny",
        }
        
        return name_map.get(model_name, model_name.replace("_", " ").title())
    
    def load_all_models(self):
        """Discover and load all models"""
        model_names = self.discover_models()
        
        for model_name in model_names:
            print(f"\n📦 Loading {model_name}...")
            self.models[model_name] = self.load_model_data(model_name)
            self.model_order.append(model_name)
        
        print(f"\n✅ Loaded {len(self.models)} models successfully\n")
    
    def create_metrics_table(self) -> pd.DataFrame:
        """Create comparison table from evaluation metrics"""
        rows = []
        
        for model_name in self.model_order:
            model = self.models[model_name]
            metrics = model.get("metrics", {})
            
            if metrics:
                row = {
                    "Model": model["display_name"],
                    "mIoU": metrics.get("miou", 0),
                    "mF1": metrics.get("mf1", 0),
                    "Accuracy": metrics.get("accuracy", 0),
                    "Clean IoU": metrics.get("iou_per_class", {}).get("Clean", 0),
                    "Damage IoU": metrics.get("iou_per_class", {}).get("Damage", 0),
                    "Best Epoch": model.get("best_epoch", "?"),
                    "Size (MB)": model.get("checkpoint_size_mb", 0),
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by mIoU descending
        df = df.sort_values("mIoU", ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_training_curves(self, output_path: Path):
        """Plot training curves: loss, mIoU, mF1"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))
        
        metrics = [
            ("train_loss", "Training Loss", axes[0, 0]),
            ("val_loss", "Validation Loss", axes[0, 1]),
            ("val_miou", "Validation mIoU", axes[1, 0]),
            ("val_mf1", "Validation mF1", axes[1, 1]),
        ]
        
        for i, model_name in enumerate(self.model_order):
            model = self.models[model_name]
            df = model.get("training_log")
            
            if df is None:
                continue
            
            color = colors[i]
            label = model["display_name"]
            
            for metric_col, title, ax in metrics:
                if metric_col in df.columns:
                    ax.plot(df["epoch"], df[metric_col], 
                           label=label, color=color, linewidth=2, alpha=0.8)
                    
                    # Mark best epoch for validation metrics
                    if metric_col.startswith("val_") and "best_epoch" in model:
                        best_epoch = model["best_epoch"]
                        best_val = df[df["epoch"] == best_epoch][metric_col].values[0]
                        ax.scatter([best_epoch], [best_val], 
                                 color=color, s=100, zorder=5, marker='*', 
                                 edgecolors='black', linewidths=1.5)
        
        # Configure axes
        for metric_col, title, ax in metrics:
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Training Curves Comparison", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved training curves: {output_path}")
    
    def plot_metrics_comparison(self, output_path: Path):
        """Create bar charts comparing key metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        model_names = [self.models[m]["display_name"] for m in self.model_order]
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))
        
        metrics_data = {
            "mIoU": [],
            "mF1": [],
            "Accuracy": [],
            "Damage IoU": [],
        }
        
        for model_name in self.model_order:
            metrics = self.models[model_name].get("metrics", {})
            metrics_data["mIoU"].append(metrics.get("miou", 0) * 100)
            metrics_data["mF1"].append(metrics.get("mf1", 0) * 100)
            metrics_data["Accuracy"].append(metrics.get("accuracy", 0) * 100)
            metrics_data["Damage IoU"].append(
                metrics.get("iou_per_class", {}).get("Damage", 0) * 100
            )
        
        # Plot bars
        plots = [
            (metrics_data["mIoU"], "mIoU (%)", axes[0, 0]),
            (metrics_data["mF1"], "mF1 (%)", axes[0, 1]),
            (metrics_data["Accuracy"], "Accuracy (%)", axes[1, 0]),
            (metrics_data["Damage IoU"], "Damage Class IoU (%)", axes[1, 1]),
        ]
        
        for values, title, ax in plots:
            bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, max(values) * 1.15])  # Add space for labels
        
        plt.suptitle("Evaluation Metrics Comparison", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved metrics comparison: {output_path}")
    
    def plot_per_class_iou(self, output_path: Path):
        """Plot per-class IoU comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        model_names = [self.models[m]["display_name"] for m in self.model_order]
        clean_ious = []
        damage_ious = []
        
        for model_name in self.model_order:
            metrics = self.models[model_name].get("metrics", {})
            iou_per_class = metrics.get("iou_per_class", {})
            clean_ious.append(iou_per_class.get("Clean", 0) * 100)
            damage_ious.append(iou_per_class.get("Damage", 0) * 100)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, clean_ious, width, label='Clean', 
                      color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, damage_ious, width, label='Damage', 
                      color='salmon', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('IoU (%)', fontsize=12)
        ax.set_title('Per-Class IoU Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(max(clean_ious), max(damage_ious)) * 1.15])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved per-class IoU: {output_path}")
    
    def plot_convergence_analysis(self, output_path: Path):
        """Analyze and plot convergence speed"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.models)))
        
        convergence_data = []
        
        for i, model_name in enumerate(self.model_order):
            model = self.models[model_name]
            df = model.get("training_log")
            
            if df is None:
                continue
            
            best_epoch = model.get("best_epoch", len(df))
            best_miou = model.get("best_val_miou", df["val_miou"].max())
            
            # Plot validation mIoU
            ax.plot(df["epoch"], df["val_miou"], 
                   label=model["display_name"], color=colors[i], 
                   linewidth=2, alpha=0.7)
            
            # Mark best epoch
            ax.scatter([best_epoch], [best_miou], 
                      color=colors[i], s=200, zorder=5, marker='*',
                      edgecolors='black', linewidths=2)
            
            # Add annotation
            ax.annotate(f'Epoch {best_epoch}\n{best_miou:.4f}',
                       xy=(best_epoch, best_miou),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', fc=colors[i], alpha=0.3))
            
            convergence_data.append({
                "Model": model["display_name"],
                "Best Epoch": best_epoch,
                "Best mIoU": best_miou,
            })
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation mIoU", fontsize=12)
        ax.set_title("Convergence Speed Analysis", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved convergence analysis: {output_path}")
        
        return pd.DataFrame(convergence_data)
    
    def export_comparison_table(self, output_path: Path):
        """Export comparison table as CSV and formatted text"""
        df = self.create_metrics_table()
        
        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved comparison table (CSV): {csv_path}")
        
        # Save formatted text
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("POC-5 MODEL COMPARISON - SUMMARY TABLE\n")
            f.write("=" * 100 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Add rankings
            f.write("=" * 100 + "\n")
            f.write("RANKINGS\n")
            f.write("=" * 100 + "\n\n")
            
            for idx, row in df.iterrows():
                medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"{idx+1}."
                f.write(f"{medal} {row['Model']}: {row['mIoU']:.4f} mIoU\n")
            
            f.write("\n")
            
            # Calculate improvements
            if len(df) >= 2:
                f.write("=" * 100 + "\n")
                f.write("PERFORMANCE IMPROVEMENTS\n")
                f.write("=" * 100 + "\n\n")
                
                best_model = df.iloc[0]
                for idx in range(1, len(df)):
                    compare_model = df.iloc[idx]
                    improvement = ((best_model['mIoU'] - compare_model['mIoU']) / compare_model['mIoU']) * 100
                    damage_improvement = ((best_model['Damage IoU'] - compare_model['Damage IoU']) / compare_model['Damage IoU']) * 100
                    
                    f.write(f"{best_model['Model']} vs {compare_model['Model']}:\n")
                    f.write(f"  - mIoU: +{improvement:.1f}%\n")
                    f.write(f"  - Damage IoU: +{damage_improvement:.1f}%\n\n")
        
        print(f"✅ Saved comparison table (TXT): {txt_path}")
        
        return df
    
    def create_summary_report(self, output_path: Path):
        """Create comprehensive summary report"""
        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("POC-5: MULTI-BACKBONE UPERNET COMPARISON - COMPREHENSIVE REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Compared: {len(self.models)}\n\n")
            
            # Overall winner
            df = self.create_metrics_table()
            winner = df.iloc[0]
            
            f.write("=" * 100 + "\n")
            f.write("🏆 WINNER\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Model: {winner['Model']}\n")
            f.write(f"mIoU: {winner['mIoU']:.4f} ({winner['mIoU']*100:.2f}%)\n")
            f.write(f"mF1: {winner['mF1']:.4f} ({winner['mF1']*100:.2f}%)\n")
            f.write(f"Accuracy: {winner['Accuracy']:.4f} ({winner['Accuracy']*100:.2f}%)\n")
            f.write(f"Damage IoU: {winner['Damage IoU']:.4f} ({winner['Damage IoU']*100:.2f}%)\n")
            f.write(f"Best Epoch: {winner['Best Epoch']}\n\n")
            
            # Detailed results per model
            f.write("=" * 100 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 100 + "\n\n")
            
            for model_name in self.model_order:
                model = self.models[model_name]
                metrics = model.get("metrics", {})
                
                f.write(f"📊 {model['display_name']}\n")
                f.write("-" * 100 + "\n")
                
                if metrics:
                    f.write(f"Overall Metrics:\n")
                    f.write(f"  - mIoU: {metrics.get('miou', 0):.4f}\n")
                    f.write(f"  - mF1: {metrics.get('mf1', 0):.4f}\n")
                    f.write(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}\n\n")
                    
                    f.write(f"Per-Class IoU:\n")
                    for class_name, iou in metrics.get('iou_per_class', {}).items():
                        f.write(f"  - {class_name}: {iou:.4f}\n")
                    
                    f.write(f"\nPer-Class F1:\n")
                    for class_name, f1 in metrics.get('f1_per_class', {}).items():
                        f.write(f"  - {class_name}: {f1:.4f}\n")
                    
                    f.write(f"\nPer-Class Precision:\n")
                    for class_name, prec in metrics.get('precision_per_class', {}).items():
                        f.write(f"  - {class_name}: {prec:.4f}\n")
                    
                    f.write(f"\nPer-Class Recall:\n")
                    for class_name, rec in metrics.get('recall_per_class', {}).items():
                        f.write(f"  - {class_name}: {rec:.4f}\n")
                
                f.write(f"\nTraining:\n")
                f.write(f"  - Best Epoch: {model.get('best_epoch', 'N/A')}\n")
                f.write(f"  - Checkpoint Size: {model.get('checkpoint_size_mb', 0):.1f} MB\n")
                f.write(f"  - Predictions Generated: {model.get('num_predictions', 0)}\n\n")
        
        print(f"✅ Saved comprehensive report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare POC-5 models")
    parser.add_argument("--logs", default="logs", help="Logs directory")
    parser.add_argument("--output", default="logs/comparison", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Initialize comparison
    comparison = ModelComparison(logs_dir=args.logs)
    
    # Load all models
    comparison.load_all_models()
    
    if len(comparison.models) == 0:
        print("❌ No models found! Check logs directory.")
        return 1
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    comparison.plot_training_curves(output_dir / "training_curves.png")
    comparison.plot_metrics_comparison(output_dir / "metrics_comparison.png")
    comparison.plot_per_class_iou(output_dir / "per_class_iou.png")
    comparison.plot_convergence_analysis(output_dir / "convergence_analysis.png")
    
    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80 + "\n")
    
    comparison.export_comparison_table(output_dir / "comparison_table")
    comparison.create_summary_report(output_dir / "summary_report.txt")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - metrics_comparison.png")
    print("  - per_class_iou.png")
    print("  - convergence_analysis.png")
    print("  - comparison_table.csv")
    print("  - comparison_table.txt")
    print("  - summary_report.txt")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
