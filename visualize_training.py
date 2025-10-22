"""
Visualize training progress from saved Tune/RLlib results (robust to different output keys)

- Finds the most recent PPO_* run under ./results/koehandel_training
- Loads result.json / result.jsonl (NDJSON) and extracts metrics with fallbacks
- Produces a multi-panel PNG with progress, losses, entropy and speed
- CLI: call directly (shows plot and saves PNG to run directory)
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt


RESULTS_ROOT = Path("./results/koehandel_training")


def find_latest_run_dir() -> Optional[Path]:
    if not RESULTS_ROOT.exists():
        return None
    ppo_dirs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith("PPO_")]
    if not ppo_dirs:
        return None
    return max(ppo_dirs, key=lambda p: p.stat().st_mtime)


def load_result_lines(result_file: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not result_file.exists():
        return items
    with result_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # If file is a single JSON array, try to parse on the fly
                try:
                    arr = json.loads(result_file.read_text(encoding="utf-8", errors="ignore"))
                    if isinstance(arr, list):
                        return arr
                except Exception:
                    continue
    return items


def safe_get(m: Dict[str, Any], keys: List[str], default=None):
    cur = m
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_metric(entry: Dict[str, Any], candidates: List[List[str]], default=0.0):
    for path in candidates:
        val = safe_get(entry, path)
        if val is not None:
            return val
    return default


def load_training_data(run_dir: Path):
    result_file = None
    for name in ("result.json", "result.jsonl", "results.json"):
        candidate = run_dir / name
        if candidate.exists():
            result_file = candidate
            break
    if result_file is None:
        json_files = list(run_dir.glob("*.json"))
        if json_files:
            result_file = max(json_files, key=lambda p: p.stat().st_mtime)
        else:
            return None

    lines = load_result_lines(result_file)
    if not lines:
        return None

    data = {
        "iterations": [],
        "timesteps": [],
        "total_loss": [],
        "policy_loss": [],
        "vf_loss": [],
        "entropy": [],
        "learning_rate": [],
        "kl_coeff": [],
        "time_elapsed": []
    }

    for entry in lines:
        data["iterations"].append(entry.get("training_iteration", entry.get("iteration", 0)))
        data["timesteps"].append(
            extract_metric(entry, [
                ["num_env_steps_sampled_lifetime"],
                ["timesteps_total"],
                ["num_env_steps_sampled"],
                ["timesteps_this_iter"]
            ], 0)
        )

        # Learner keys: new RLlib often uses 'learners' -> policy_id -> metric_name
        # Try multiple fallbacks
        data["total_loss"].append(
            extract_metric(entry, [
                ["learners", "shared_policy", "total_loss"],
                ["learners", "player_0", "total_loss"],
                ["info", "learner", "default_policy", "learner_stats", "total_loss"],
                ["info", "learner", "default_policy", "learner_stats", "policy_loss"]
            ], 0.0)
        )
        data["policy_loss"].append(
            extract_metric(entry, [
                ["learners", "shared_policy", "policy_loss"],
                ["learners", "player_0", "policy_loss"],
                ["info", "learner", "default_policy", "learner_stats", "policy_loss"]
            ], 0.0)
        )
        data["vf_loss"].append(
            extract_metric(entry, [
                ["learners", "shared_policy", "vf_loss"],
                ["learners", "player_0", "vf_loss"],
                ["info", "learner", "default_policy", "learner_stats", "vf_loss"]
            ], 0.0)
        )
        data["entropy"].append(
            extract_metric(entry, [
                ["learners", "shared_policy", "entropy"],
                ["learners", "player_0", "entropy"],
                ["info", "learner", "default_policy", "learner_stats", "entropy"]
            ], 0.0)
        )
        data["learning_rate"].append(
            extract_metric(entry, [
                ["learners", "shared_policy", "default_optimizer_learning_rate"],
                ["learners", "player_0", "default_optimizer_learning_rate"],
                ["info", "learner", "default_policy", "learner_stats", "default_optimizer_learning_rate"]
            ], 0.0)
        )
        data["kl_coeff"].append(
            extract_metric(entry, [
                ["learners", "shared_policy", "curr_kl_coeff"],
                ["learners", "player_0", "curr_kl_coeff"],
                ["info", "learner", "default_policy", "learner_stats", "curr_kl_coeff"]
            ], 0.0)
        )
        data["time_elapsed"].append(entry.get("time_total_s", entry.get("time_this_iter_s", 0.0)))

    # Convert lists to numpy arrays for plotting convenience
    for k in list(data.keys()):
        data[k] = np.array(data[k], dtype=float)

    return data


def plot_training_progress(data: Dict[str, np.ndarray], run_name: str, save_dir: Path):
    iterations = data["iterations"]
    timesteps = data["timesteps"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Koehandel Training Progress - {run_name}", fontsize=16, fontweight="bold")

    # 1: Timesteps vs Iterations
    ax = axes[0, 0]
    ax.plot(iterations, timesteps, color="tab:blue", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Env Steps")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y")
    if len(timesteps) > 0:
        target = 1_000_000
        current = timesteps[-1]
        progress_pct = (current / target) * 100
        ax.axhline(y=target, color="r", linestyle="--", alpha=0.5, label="Target (1M)")
        ax.text(0.95, 0.95, f"{progress_pct:.1f}% Complete\n{int(current):,} / {target:,} steps",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()

    # 2: Total Loss
    ax = axes[0, 1]
    ax.plot(iterations, data["total_loss"], "r-", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss Over Time")
    ax.grid(True, alpha=0.3)
    if len(data["total_loss"]) > 10:
        window = min(20, max(3, len(data["total_loss"]) // 5))
        smoothed = np.convolve(data["total_loss"], np.ones(window) / window, mode="valid")
        ax.plot(iterations[window - 1 :], smoothed, "k--", linewidth=1.5, alpha=0.7, label="Smoothed")
        ax.legend()

    # 3: Policy and Value Loss
    ax = axes[1, 0]
    ax.plot(iterations, data["policy_loss"], label="Policy Loss", color="tab:blue", alpha=0.8)
    ax.plot(iterations, data["vf_loss"], label="Value Loss", color="tab:green", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Policy and Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4: Entropy
    ax = axes[1, 1]
    ax.plot(iterations, data["entropy"], color="purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy (Exploration)")
    ax.grid(True, alpha=0.3)
    if len(data["entropy"]) > 0:
        avg_entropy = np.mean(data["entropy"][-20:]) if len(data["entropy"]) >= 1 else data["entropy"][-1]
        ax.text(0.95, 0.95, f"Recent Avg: {avg_entropy:.3f}", transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.5))

    # 5: Learning Rate and KL
    ax = axes[2, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(iterations, data["learning_rate"], "b-", label="Learning Rate")
    l2 = ax2.plot(iterations, data["kl_coeff"], "r-", label="KL Coeff")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate", color="b")
    ax2.set_ylabel("KL Coeff", color="r")
    ax.set_title("Learning Rate and KL Control")
    ax.grid(True, alpha=0.3)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right")

    # 6: Training Speed
    ax = axes[2, 1]
    if len(data["time_elapsed"]) > 1 and len(timesteps) > 1:
        times_min = data["time_elapsed"] / 60.0
        steps_per_min = timesteps / (times_min + 1e-6)
        ax.plot(iterations, steps_per_min, color="tab:orange")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Steps per Minute")
        ax.set_title("Training Speed")
        ax.grid(True, alpha=0.3)
        if len(steps_per_min) > 0:
            avg_speed = np.mean(steps_per_min[-20:])
            ax.axhline(avg_speed, color="r", linestyle="--", alpha=0.5)
            ax.text(0.95, 0.95, f"Recent Avg:\n{avg_speed:.0f} steps/min", transform=ax.transAxes,
                    ha="right", va="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))
            if timesteps[-1] > 0 and avg_speed > 0:
                remaining_steps = max(0.0, 1_000_000 - timesteps[-1])
                eta_min = remaining_steps / avg_speed
                ax.text(0.05, 0.05, f"ETA: {eta_min:.0f} min", transform=ax.transAxes, ha="left", va="bottom",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"training_progress_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] Plot saved to: {save_path}")
    return fig


def print_summary_stats(data: Dict[str, np.ndarray], run_name: str):
    print("\n" + "=" * 70)
    print(f"TRAINING SUMMARY - {run_name}")
    print("=" * 70)
    if data is None or len(data["iterations"]) == 0:
        print("[INFO] No data available yet")
        return

    current_iter = int(data["iterations"][-1])
    current_steps = int(data["timesteps"][-1])
    time_min = float(data["time_elapsed"][-1]) / 60.0 if data["time_elapsed"].size > 0 else 0.0
    print(f"\n[PROGRESS]")
    print(f"  Current Iteration:  {current_iter:,}")
    print(f"  Environment Steps:  {current_steps:,.0f} / 1,000,000")
    print(f"  Progress:           {(current_steps / 1_000_000) * 100:.1f}%")
    print(f"  Time Elapsed:       {time_min:.1f} minutes")

    if time_min > 0:
        steps_per_min = current_steps / time_min
        print(f"  Training Speed:     {steps_per_min:.0f} steps/min")
        remaining_steps = 1_000_000 - current_steps
        eta_min = remaining_steps / (steps_per_min + 1e-6)
        print(f"  ETA to completion:  {eta_min:.0f} minutes")

    print(f"\n[RECENT PERFORMANCE (Last 20 iterations)]")
    def mean_safe(arr):
        return float(np.mean(arr[-20:])) if arr.size > 0 else 0.0

    print(f"  Avg Total Loss:     {mean_safe(data['total_loss']):.4f}")
    print(f"  Avg Policy Loss:    {mean_safe(data['policy_loss']):.4f}")
    print(f"  Avg Value Loss:     {mean_safe(data['vf_loss']):.4f}")
    print(f"  Avg Entropy:        {mean_safe(data['entropy']):.4f}")

    if data["total_loss"].size > 40:
        early_loss = float(np.mean(data["total_loss"][10:30]))
        recent_loss = float(np.mean(data["total_loss"][-20:]))
        improvement = ((early_loss - recent_loss) / (early_loss + 1e-9)) * 100.0
        print("\n[LEARNING PROGRESS]")
        print(f"  Early Loss (iter 10-30):  {early_loss:.4f}")
        print(f"  Recent Loss:              {recent_loss:.4f}")
        print(f"  Improvement:              {improvement:+.1f}%")
        if improvement > 0:
            print("  Status: [OK] Model is learning!")
        else:
            print("  Status: [NOTE] Loss not decreasing yet")
    print("=" * 70)


def visualize_latest_training():
    run_dir = find_latest_run_dir()
    if run_dir is None:
        print("[ERROR] No training run directories found under ./results/koehandel_training")
        return

    print("=" * 70)
    print("KOEHANDEL TRAINING VISUALIZATION")
    print("=" * 70)
    print(f"Analyzing: {run_dir.name}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data = load_training_data(run_dir)
    if data is None:
        print("[ERROR] Could not load training data from run directory")
        return

    print(f"[OK] Loaded {len(data['iterations'])} iterations of data")

    print("\n[Computing statistics...]")
    print_summary_stats(data, run_dir.name)

    print("\n[Creating visualizations...]")
    fig = plot_training_progress(data, run_dir.name, run_dir)

    print("\n[SUCCESS] Visualization complete!")
    print(f"Saved plot to run directory: {run_dir}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    visualize_latest_training()