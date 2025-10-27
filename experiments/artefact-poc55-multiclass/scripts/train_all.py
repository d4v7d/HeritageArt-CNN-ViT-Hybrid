"""
POC-5.5 Batch Training Script
Train all 3 models sequentially with monitoring.

Usage:
    python train_all.py [--output-dir logs] [--skip MODEL1,MODEL2]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time


class BatchTrainer:
    """Batch training manager for POC-5.5"""
    
    MODELS = {
        'convnext': {
            'name': 'ConvNeXt-Tiny',
            'config': 'configs/convnext_tiny.yaml',
            'output': 'logs/convnext_tiny',
            'estimated_time_min': 46,
        },
        'swin': {
            'name': 'Swin-Tiny',
            'config': 'configs/swin_tiny.yaml',
            'output': 'logs/swin_tiny',
            'estimated_time_min': 46,
        },
        'maxvit': {
            'name': 'MaxViT-Tiny',
            'config': 'configs/maxvit_tiny.yaml',
            'output': 'logs/maxvit_tiny',
            'estimated_time_min': 46,
        },
    }
    
    def __init__(self, output_dir: str = 'logs', skip_models: list = None):
        self.output_dir = Path(output_dir)
        self.skip_models = skip_models or []
        self.results = {}
        
    def train_model(self, model_key: str) -> bool:
        """Train a single model"""
        model = self.MODELS[model_key]
        
        print("\n" + "=" * 80)
        print(f"🚂 TRAINING {model['name'].upper()}")
        print("=" * 80)
        print(f"Config: {model['config']}")
        print(f"Output: {model['output']}")
        print(f"Estimated time: {model['estimated_time_min']} minutes")
        
        # Calculate ETA
        start_time = datetime.now()
        eta = start_time + timedelta(minutes=model['estimated_time_min'])
        print(f"Start: {start_time.strftime('%H:%M:%S')}")
        print(f"ETA:   {eta.strftime('%H:%M:%S')}")
        print("-" * 80 + "\n")
        
        # Build command
        cmd = [
            sys.executable,
            "scripts/train_poc55.py",
            "--config", model['config'],
            "--output-dir", str(self.output_dir),
        ]
        
        # Run training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            elapsed = datetime.now() - start_time
            elapsed_min = elapsed.total_seconds() / 60
            
            print("\n" + "-" * 80)
            print(f"✅ {model['name']} training completed!")
            print(f"Elapsed time: {elapsed_min:.1f} minutes")
            print(f"Checkpoint saved to: {model['output']}/checkpoints/best_model.pth")
            print("=" * 80)
            
            self.results[model_key] = {
                'status': 'success',
                'elapsed_min': elapsed_min,
                'start': start_time,
                'end': datetime.now(),
            }
            
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = datetime.now() - start_time
            elapsed_min = elapsed.total_seconds() / 60
            
            print("\n" + "-" * 80)
            print(f"❌ {model['name']} training failed!")
            print(f"Elapsed time: {elapsed_min:.1f} minutes")
            print(f"Error: {e}")
            print("=" * 80)
            
            self.results[model_key] = {
                'status': 'failed',
                'elapsed_min': elapsed_min,
                'error': str(e),
                'start': start_time,
                'end': datetime.now(),
            }
            
            return False
    
    def run_all(self):
        """Train all models sequentially"""
        models_to_train = [k for k in self.MODELS.keys() if k not in self.skip_models]
        
        print("\n" + "=" * 80)
        print("🚂 POC-5.5 BATCH TRAINING")
        print("=" * 80)
        print(f"Models to train: {len(models_to_train)}")
        for key in models_to_train:
            print(f"  - {self.MODELS[key]['name']}")
        
        total_time = sum(self.MODELS[k]['estimated_time_min'] for k in models_to_train)
        print(f"\nTotal estimated time: {total_time} minutes (~{total_time/60:.1f} hours)")
        
        start_time = datetime.now()
        eta = start_time + timedelta(minutes=total_time)
        print(f"Start: {start_time.strftime('%H:%M:%S')}")
        print(f"ETA:   {eta.strftime('%H:%M:%S')}")
        print("=" * 80)
        
        # Confirmation
        response = input("\n⚠️  This will take ~2.3 hours. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Cancelled by user")
            return
        
        # Train each model
        for i, model_key in enumerate(models_to_train, 1):
            print(f"\n\n{'#' * 80}")
            print(f"# MODEL {i}/{len(models_to_train)}: {self.MODELS[model_key]['name']}")
            print(f"{'#' * 80}\n")
            
            success = self.train_model(model_key)
            
            if not success:
                print(f"\n⚠️  Model {model_key} failed. Continue with next model? [y/N]: ")
                response = input()
                if response.lower() not in ['y', 'yes']:
                    print("❌ Batch training stopped")
                    break
        
        # Final summary
        self.print_summary(start_time)
    
    def print_summary(self, start_time: datetime):
        """Print training summary"""
        end_time = datetime.now()
        total_elapsed = end_time - start_time
        total_elapsed_min = total_elapsed.total_seconds() / 60
        
        print("\n\n" + "=" * 80)
        print("📊 BATCH TRAINING SUMMARY")
        print("=" * 80)
        
        print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_elapsed_min:.1f} minutes ({total_elapsed_min/60:.2f} hours)")
        
        print("\n" + "-" * 80)
        print("MODEL RESULTS:")
        print("-" * 80)
        
        success_count = 0
        for model_key, result in self.results.items():
            model_name = self.MODELS[model_key]['name']
            status_icon = "✅" if result['status'] == 'success' else "❌"
            
            print(f"\n{status_icon} {model_name}:")
            print(f"   Status: {result['status']}")
            print(f"   Time: {result['elapsed_min']:.1f} minutes")
            
            if result['status'] == 'success':
                success_count += 1
                checkpoint = self.MODELS[model_key]['output'] + '/checkpoints/best_model.pth'
                print(f"   Checkpoint: {checkpoint}")
            else:
                print(f"   Error: {result.get('error', 'Unknown')}")
        
        print("\n" + "=" * 80)
        print(f"✅ Successful: {success_count}/{len(self.results)}")
        print(f"❌ Failed:     {len(self.results) - success_count}/{len(self.results)}")
        print("=" * 80)
        
        if success_count == len(self.results):
            print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
            print("\nNext steps:")
            print("  1. Evaluate all models: make eval-all")
            print("  2. Compare results:     make compare")
            print("  3. Check reports:       logs/comparison/summary_report.txt")
        elif success_count > 0:
            print("\n⚠️  Some models trained successfully")
            print("   Review errors above and retry failed models")
        else:
            print("\n❌ ALL MODELS FAILED")
            print("   Check error messages above")
        
        print()


def main():
    parser = argparse.ArgumentParser(description="POC-5.5 Batch Training")
    parser.add_argument("--output-dir", default="logs", help="Output directory")
    parser.add_argument("--skip", default="", help="Comma-separated models to skip (convnext,swin,maxvit)")
    args = parser.parse_args()
    
    # Parse skip list
    skip_models = [m.strip() for m in args.skip.split(",") if m.strip()]
    
    # Run batch training
    trainer = BatchTrainer(
        output_dir=args.output_dir,
        skip_models=skip_models
    )
    
    trainer.run_all()


if __name__ == "__main__":
    main()
