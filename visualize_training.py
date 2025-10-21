"""
Visualize training progress from saved results
Creates graphs showing loss, rewards, and learning curves
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


def load_training_data(run_dir):
    """Load all training metrics from result.json."""
    result_file = run_dir / "result.json"

    if not result_file.exists():
        return None

    data = {
        'iterations': [],
        'timesteps': [],
        'total_loss': [],
        'policy_loss': [],
        'vf_loss': [],
        'entropy': [],
        'learning_rate': [],
        'kl_coeff': [],
        'time_elapsed': []
    }

    with open(result_file, 'r') as f:
        for line in f:
            try:
                metrics = json.loads(line)

                data['iterations'].append(metrics.get('training_iteration', 0))
                data['timesteps'].append(metrics.get('num_env_steps_sampled_lifetime', 0))
                data['total_loss'].append(metrics.get('learners/player_0/total_loss', 0))
                data['policy_loss'].append(metrics.get('learners/player_0/policy_loss', 0))
                data['vf_loss'].append(metrics.get('learners/player_0/vf_loss', 0))
                data['entropy'].append(metrics.get('learners/player_0/entropy', 0))
                data['learning_rate'].append(metrics.get('learners/player_0/default_optimizer_learning_rate', 0))
                data['kl_coeff'].append(metrics.get('learners/player_0/curr_kl_coeff', 0))
                data['time_elapsed'].append(metrics.get('time_total_s', 0))

            except json.JSONDecodeError:
                continue

    return data


def plot_training_progress(data, run_name, save_dir):
    """Create comprehensive training progress plots."""

    # Convert to numpy arrays
    iterations = np.array(data['iterations'])
    timesteps = np.array(data['timesteps'])

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Koehandel Training Progress - {run_name}', fontsize=16, fontweight='bold')

    # 1. Training Progress (Timesteps vs Iterations)
    ax = axes[0, 0]
    ax.plot(iterations, timesteps, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Environment Steps')
    ax.set_title('Training Progress')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')

    # Add progress percentage
    if len(timesteps) > 0:
        target = 1_000_000
        current = timesteps[-1]
        progress_pct = (current / target) * 100
        ax.axhline(y=target, color='r', linestyle='--', alpha=0.5, label=f'Target (1M)')
        ax.text(0.95, 0.95, f'{progress_pct:.1f}% Complete\n{current:,.0f} / {target:,} steps',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()

    # 2. Total Loss
    ax = axes[0, 1]
    ax.plot(iterations, data['total_loss'], 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Over Time')
    ax.grid(True, alpha=0.3)

    # Add smoothed trend line
    if len(data['total_loss']) > 10:
        window = min(20, len(data['total_loss']) // 5)
        smoothed = np.convolve(data['total_loss'], np.ones(window) / window, mode='valid')
        ax.plot(iterations[window - 1:], smoothed, 'k--', linewidth=2, alpha=0.7, label='Smoothed')
        ax.legend()

    # 3. Policy and Value Loss
    ax = axes[1, 0]
    ax.plot(iterations, data['policy_loss'], 'b-', linewidth=2, label='Policy Loss', alpha=0.7)
    ax.plot(iterations, data['vf_loss'], 'g-', linewidth=2, label='Value Loss', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Policy and Value Function Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Entropy
    ax = axes[1, 1]
    ax.plot(iterations, data['entropy'], 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy (Exploration)')
    ax.grid(True, alpha=0.3)

    # Add note about entropy
    if len(data['entropy']) > 0:
        avg_entropy = np.mean(data['entropy'][-20:])
        ax.text(0.95, 0.95, f'Recent Avg: {avg_entropy:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    # 5. Learning Rate and KL Coefficient
    ax = axes[2, 0]
    ax2 = ax.twinx()

    line1 = ax.plot(iterations, data['learning_rate'], 'b-', linewidth=2, label='Learning Rate')
    line2 = ax2.plot(iterations, data['kl_coeff'], 'r-', linewidth=2, label='KL Coefficient')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate', color='b')
    ax2.set_ylabel('KL Coefficient', color='r')
    ax.set_title('Learning Rate and KL Divergence Control')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')

    # 6. Training Speed
    ax = axes[2, 1]
    if len(data['time_elapsed']) > 1:
        time_elapsed_minutes = np.array(data['time_elapsed']) / 60
        steps_per_minute = timesteps / (time_elapsed_minutes + 0.001)  # Avoid division by zero

        ax.plot(iterations, steps_per_minute, 'orange', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Steps per Minute')
        ax.set_title('Training Speed')
        ax.grid(True, alpha=0.3)

        # Add average speed
        if len(steps_per_minute) > 0:
            avg_speed = np.mean(steps_per_minute[-20:])
            ax.axhline(y=avg_speed, color='r', linestyle='--', alpha=0.5)
            ax.text(0.95, 0.95, f'Recent Avg:\n{avg_speed:.0f} steps/min',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            # Estimate time remaining
            if timesteps[-1] > 0:
                remaining_steps = 1_000_000 - timesteps[-1]
                eta_minutes = remaining_steps / avg_speed
                ax.text(0.05, 0.05, f'ETA: {eta_minutes:.0f} min',
                        transform=ax.transAxes, ha='left', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = save_dir / f'training_progress_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Plot saved to: {save_path}")

    return fig


def print_summary_stats(data, run_name):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print(f"TRAINING SUMMARY - {run_name}")
    print("=" * 70)

    if not data['iterations']:
        print("[INFO] No data available yet")
        return

    current_iter = data['iterations'][-1]
    current_steps = data['timesteps'][-1]
    current_time = data['time_elapsed'][-1] / 60  # minutes

    print(f"\n[PROGRESS]")
    print(f"  Current Iteration:  {current_iter:,}")
    print(f"  Environment Steps:  {current_steps:,.0f} / 1,000,000")
    print(f"  Progress:           {(current_steps / 1_000_000) * 100:.1f}%")
    print(f"  Time Elapsed:       {current_time:.1f} minutes")

    if current_time > 0:
        steps_per_min = current_steps / current_time
        print(f"  Training Speed:     {steps_per_min:.0f} steps/min")

        remaining_steps = 1_000_000 - current_steps
        eta_min = remaining_steps / steps_per_min
        print(f"  ETA to completion:  {eta_min:.0f} minutes")

    print(f"\n[RECENT PERFORMANCE (Last 20 iterations)]")
    recent_total_loss = np.mean(data['total_loss'][-20:])
    recent_policy_loss = np.mean(data['policy_loss'][-20:])
    recent_vf_loss = np.mean(data['vf_loss'][-20:])
    recent_entropy = np.mean(data['entropy'][-20:])

    print(f"  Avg Total Loss:     {recent_total_loss:.4f}")
    print(f"  Avg Policy Loss:    {recent_policy_loss:.4f}")
    print(f"  Avg Value Loss:     {recent_vf_loss:.4f}")
    print(f"  Avg Entropy:        {recent_entropy:.4f}")

    # Check if losses are decreasing (sign of learning)
    if len(data['total_loss']) > 40:
        early_loss = np.mean(data['total_loss'][10:30])
        recent_loss = np.mean(data['total_loss'][-20:])
        improvement = ((early_loss - recent_loss) / early_loss) * 100

        print(f"\n[LEARNING PROGRESS]")
        print(f"  Early Loss (iter 10-30):  {early_loss:.4f}")
        print(f"  Recent Loss:              {recent_loss:.4f}")
        print(f"  Improvement:              {improvement:+.1f}%")

        if improvement > 0:
            print("  Status: [OK] Model is learning!")
        else:
            print("  Status: [NOTE] Loss not decreasing yet")

    print("=" * 70)


def visualize_latest_training():
    """Find and visualize the most recent training run."""
    results_dir = Path("./results/koehandel_training")

    if not results_dir.exists():
        print("[ERROR] No training results found")
        print(f"        Expected directory: {results_dir.absolute()}")
        return

    # Find all PPO directories
    ppo_dirs = list(results_dir.glob("PPO_*"))

    if not ppo_dirs:
        print("[ERROR] No training runs found")
        return

    # Get the most recent one
    latest_dir = max(ppo_dirs, key=lambda p: p.stat().st_mtime)

    print("=" * 70)
    print("KOEHANDEL TRAINING VISUALIZATION")
    print("=" * 70)
    print(f"Analyzing: {latest_dir.name}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading training data...")
    data = load_training_data(latest_dir)

    if data is None or not data['iterations']:
        print("[ERROR] Could not load training data")
        return

    print(f"[OK] Loaded {len(data['iterations'])} iterations of data")

    # Print summary
    print("\n[2/3] Computing statistics...")
    print_summary_stats(data, latest_dir.name)

    # Create visualizations
    print("\n[3/3] Creating visualizations...")
    fig = plot_training_progress(data, latest_dir.name, latest_dir)

    print("\n[SUCCESS] Visualization complete!")
    print("\nTo view the plot:")
    print("  1. Check the saved PNG file in the results directory")
    print("  2. Or run this script again to see updated progress")
    print("\nPress Enter to close (or Ctrl+C)...")

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nClosed by user")


if __name__ == "__main__":
    try:
        import matplotlib

        matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
    except:
        pass

    visualize_latest_training()