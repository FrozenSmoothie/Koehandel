"""
Quick training status check - no continuous monitoring (NEW API compatible)
"""

import json
from pathlib import Path
from datetime import datetime

def check_status():
    """Check current training status."""
    results_dir = Path("./results/koehandel_training")

    if not results_dir.exists():
        print("[INFO] No training results found")
        print("       Training has not been started yet")
        return

    # Find all PPO directories
    ppo_dirs = list(results_dir.glob("PPO_*"))

    if not ppo_dirs:
        print("[INFO] No training runs found")
        return

    print("="*70)
    print("TRAINING STATUS (NEW API)")
    print("="*70)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    for ppo_dir in sorted(ppo_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        result_file = ppo_dir / "result.json"

        if not result_file.exists():
            continue

        print(f"\nRun: {ppo_dir.name}")

        # Read last line
        with open(result_file, 'r') as f:
            lines = f.readlines()
            if lines:
                metrics = json.loads(lines[-1])

                iteration = metrics.get("training_iteration", 0)

                # FIXED: Correct metric name
                timesteps = metrics.get("num_env_steps_sampled_lifetime", 0)

                progress = (timesteps / 1_000_000) * 100 if timesteps > 0 else 0

                print(f"  Iteration:       {iteration:,}")
                print(f"  Env Steps:       {timesteps:,.0f} / 1,000,000 ({progress:.1f}%)")

                # Show losses
                total_loss = metrics.get("learners/player_0/total_loss", None)
                if total_loss is not None:
                    print(f"  Total Loss:      {total_loss:.4f}")

                policy_loss = metrics.get("learners/player_0/policy_loss", None)
                if policy_loss is not None:
                    print(f"  Policy Loss:     {policy_loss:.4f}")

                # Check for checkpoints
                checkpoints = list(ppo_dir.glob("checkpoint_*"))
                if checkpoints:
                    latest_cp = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    print(f"  Latest Checkpoint: {latest_cp.name}")

        print("-"*70)

    print("\nTo monitor in real-time, run:")
    print("  python monitor_training.py")
    print("="*70)

if __name__ == "__main__":
    check_status()