"""
Monitor training (compatible with OLD API)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

def find_latest_result_file():
    results_dir = Path("./results/koehandel_training")
    if not results_dir.exists():
        return None

    ppo_dirs = list(results_dir.glob("PPO_*"))
    if not ppo_dirs:
        return None

    latest_dir = max(ppo_dirs, key=lambda p: p.stat().st_mtime)
    result_file = latest_dir / "result.json"

    if result_file.exists():
        return result_file
    return None

def read_latest_metrics(result_file):
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
            if lines:
                return json.loads(lines[-1])
    except Exception as e:
        print(f"Error: {e}")
    return None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def monitor_training(refresh_interval=10):
    print("="*70)
    print("KOEHANDEL TRAINING MONITOR (OLD API)")
    print("="*70)

    start_time = None
    last_metrics = None

    try:
        while True:
            result_file = find_latest_result_file()

            if result_file is None:
                print("\nWaiting for training...")
                time.sleep(refresh_interval)
                continue

            metrics = read_latest_metrics(result_file)
            if metrics is None:
                time.sleep(refresh_interval)
                continue

            if start_time is None:
                start_time = metrics.get("timestamp", time.time())

            os.system('cls' if os.name == 'nt' else 'clear')

            print("="*70)
            print("KOEHANDEL TRAINING MONITOR")
            print("="*70)
            print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*70)

            current_time = metrics.get("timestamp", time.time())
            elapsed = current_time - start_time

            iteration = metrics.get("training_iteration", 0)
            timesteps = metrics.get("timesteps_total", 0)
            episodes = metrics.get("episodes_total", 0)
            reward_mean = metrics.get("episode_reward_mean", 0)
            episode_len = metrics.get("episode_len_mean", 0)

            target = 1_000_000
            progress = (timesteps / target) * 100

            print("\n[PROGRESS]")
            print(f"  Elapsed:     {format_time(elapsed)}")
            print(f"  Iteration:   {iteration:,}")
            print(f"  Timesteps:   {timesteps:,} / {target:,}")
            print(f"  Progress:    {progress:.1f}%")

            bar_length = 50
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  [{bar}] {progress:.1f}%")

            print("\n[PERFORMANCE]")
            print(f"  Episodes:    {episodes:,}")
            print(f"  Avg Reward:  {reward_mean:.4f}")
            print(f"  Avg Length:  {episode_len:.1f}")

            # Policy metrics (OLD API location)
            policy_loss = metrics.get("info/learner/default_policy/learner_stats/policy_loss", 0)
            vf_loss = metrics.get("info/learner/default_policy/learner_stats/vf_loss", 0)
            entropy = metrics.get("info/learner/default_policy/learner_stats/entropy", 0)

            if policy_loss != 0 or vf_loss != 0:
                print("\n[LEARNING]")
                print(f"  Policy Loss: {policy_loss:.4f}")
                print(f"  Value Loss:  {vf_loss:.4f}")
                print(f"  Entropy:     {entropy:.4f}")

            if timesteps > 0 and elapsed > 0:
                rate = timesteps / elapsed
                remaining = (target - timesteps) / rate if rate > 0 else 0

                print(f"\n[ETA]")
                print(f"  Steps/sec:   {rate:.1f}")
                print(f"  Remaining:   {format_time(remaining)}")

            print("\n" + "="*70)
            print(f"Refreshing in {refresh_interval}s... (Ctrl+C to stop)")
            print("="*70)

            if last_metrics:
                delta = timesteps - last_metrics.get("timesteps_total", 0)
                if delta > 0:
                    print(f"\n[DELTA] +{delta:,} steps")

            last_metrics = metrics
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

if __name__ == "__main__":
    monitor_training()