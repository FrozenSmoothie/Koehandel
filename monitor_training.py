"""
Monitor training (compatible with Tune / RLlib result.json lines output)

This updated monitor:
- Finds the most recent PPO_* run folder under ./results/koehandel_training
  and reads its result.json (NDJSON) or result.jsonl file.
- Robustly extracts metrics with multiple fallback keys used by different RLlib/Tune versions.
- Shows progress, ETA, simple learning stats and deltas between updates.
- Works on Windows and *nix (clears console appropriately).
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


RESULTS_ROOT = Path("./results/koehandel_training")


def find_latest_result_file() -> Optional[Path]:
    if not RESULTS_ROOT.exists():
        return None

    ppo_dirs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith("PPO_")]
    if not ppo_dirs:
        return None

    latest_dir = max(ppo_dirs, key=lambda p: p.stat().st_mtime)
    # Common names used by Tune: result.json, result.jsonl, results.json
    for name in ("result.json", "result.jsonl", "results.json"):
        candidate = latest_dir / name
        if candidate.exists():
            return candidate
    # fallback: search for any .json file
    json_files = list(latest_dir.glob("*.json"))
    if json_files:
        return max(json_files, key=lambda p: p.stat().st_mtime)
    return None


def read_latest_metrics(result_file: Path) -> Optional[Dict[str, Any]]:
    try:
        text = result_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return None
        # If file is NDJSON (one JSON object per line), take last non-empty line
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return None
        last = lines[-1]
        # last line may be a plain JSON object
        try:
            return json.loads(last)
        except json.JSONDecodeError:
            # maybe the whole file is a JSON array
            try:
                arr = json.loads(text)
                if isinstance(arr, list) and arr:
                    return arr[-1]
            except Exception:
                return None
    except Exception as e:
        print(f"Error reading metrics file {result_file}: {e}")
    return None


def get_nested(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def choose_metric(metrics: Dict[str, Any], candidates: List[List[str]], default=0):
    for path in candidates:
        val = get_nested(metrics, path, None)
        if val is not None:
            return val
    return default


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def monitor_training(refresh_interval: int = 10):
    print("=" * 70)
    print("KOEHANDEL TRAINING MONITOR (TUNE / RLlib compatible)")
    print("=" * 70)

    start_time = None
    last_metrics = None
    last_timesteps = 0

    try:
        while True:
            result_file = find_latest_result_file()

            if result_file is None:
                print("\nWaiting for training results in ./results/koehandel_training ...")
                time.sleep(refresh_interval)
                continue

            metrics = read_latest_metrics(result_file)
            if metrics is None:
                time.sleep(refresh_interval)
                continue

            if start_time is None:
                start_time = metrics.get("time_total_s", metrics.get("timestamp", time.time()))

            # clear console
            os.system("cls" if os.name == "nt" else "clear")

            print("=" * 70)
            print("KOEHANDEL TRAINING MONITOR")
            print("=" * 70)
            print(f"Results dir: {result_file.parent}")
            print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            current_time = metrics.get("time_total_s", metrics.get("timestamp", time.time()))
            elapsed = current_time - start_time if start_time else 0.0

            iteration = choose_metric(metrics, [["training_iteration"], ["iteration"], ["iter"]], 0)
            # timesteps: try several keys used across versions
            timesteps = choose_metric(
                metrics,
                [
                    ["num_env_steps_sampled_lifetime"],
                    ["timesteps_total"],
                    ["timesteps_this_iter"],
                    ["num_env_steps_sampled"]
                ],
                0
            )
            episodes = choose_metric(metrics, [["episodes_total"], ["episodes_this_iter"], ["episodes"]], 0)
            reward_mean = choose_metric(metrics, [["episode_reward_mean"], ["episode_reward_mean_0"], ["policy_reward_mean"]], 0.0)
            episode_len = choose_metric(metrics, [["episode_len_mean"], ["episode_length_mean"], ["episode_len"]], 0.0)

            target = 1_000_000
            progress = (timesteps / target) * 100 if target > 0 else 0.0

            print("\n[PROGRESS]")
            print(f"  Elapsed:     {format_time(elapsed)}")
            print(f"  Iteration:   {int(iteration):,}")
            print(f"  Timesteps:   {int(timesteps):,} / {target:,}")
            print(f"  Progress:    {progress:.1f}%")

            bar_length = 50
            filled = int(bar_length * min(progress, 100.0) / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  [{bar}] {progress:.1f}%")

            print("\n[PERFORMANCE]")
            print(f"  Episodes:    {int(episodes):,}")
            print(f"  Avg Reward:  {float(reward_mean):.4f}")
            print(f"  Avg Length:  {float(episode_len):.1f}")

            # Learner stats: try multiple possible locations
            # Old API: info/learner/default_policy/learner_stats/policy_loss
            # New API: learners -> {policy_id: {...}}
            policy_loss = choose_metric(metrics,
                                        [["info", "learner", "default_policy", "learner_stats", "policy_loss"],
                                         ["info", "learner", "default_policy", "learner_stats", "total_loss"],
                                         ["learners", "shared_policy", "policy_loss"],
                                         ["learners", "player_0", "policy_loss"]],
                                        0)
            vf_loss = choose_metric(metrics,
                                    [["info", "learner", "default_policy", "learner_stats", "vf_loss"],
                                     ["learners", "shared_policy", "vf_loss"],
                                     ["learners", "player_0", "vf_loss"]],
                                    0)
            entropy = choose_metric(metrics,
                                    [["info", "learner", "default_policy", "learner_stats", "entropy"],
                                     ["learners", "shared_policy", "entropy"],
                                     ["learners", "player_0", "entropy"]],
                                    0)

            if any(v != 0 for v in (policy_loss, vf_loss, entropy)):
                print("\n[LEARNING]")
                print(f"  Policy Loss: {float(policy_loss):.4f}")
                print(f"  Value Loss:  {float(vf_loss):.4f}")
                print(f"  Entropy:     {float(entropy):.4f}")

            if timesteps > 0 and elapsed > 0:
                rate = timesteps / elapsed
                remaining = (target - timesteps) / rate if rate > 0 else 0.0
                print(f"\n[ETA]")
                print(f"  Steps/sec:   {rate:.1f}")
                print(f"  Remaining:   {format_time(remaining)}")

            print("\n" + "=" * 70)
            print(f"Refreshing in {refresh_interval}s... (Ctrl+C to stop)")
            print("=" * 70)

            if last_metrics:
                delta = int(timesteps) - int(last_metrics.get("timesteps_total", last_metrics.get("num_env_steps_sampled_lifetime", 0)))
                if delta > 0:
                    print(f"\n[DELTA] +{delta:,} steps since last check")

            last_metrics = metrics
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nStopped by user")


if __name__ == "__main__":
    monitor_training()