"""
Visualize a recorded replay JSONL (saved by play_and_record.py).

Usage:
  python visualize_replay.py replays/game_YYYYMMDD_HHMMSS.jsonl

Output:
  ./replays/game_YYYYMMDD_HHMMSS_progress.png
"""
import sys
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_replay(path):
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    data = [json.loads(l) for l in lines if l.strip()]
    return data

def visualize(replay_path):
    data = load_replay(replay_path)
    if not data:
        print("No data in replay.")
        return

    # Group by turn and agent
    turns = sorted({int(d["turn"]) for d in data})
    agents = sorted(list({d["agent"] for d in data}))
    agent_idx = {a:i for i,a in enumerate(agents)}

    # Per-turn arrays
    turn_to_row = {t:i for i,t in enumerate(turns)}
    progress_grid = np.full((len(turns), len(agents)), np.nan)
    diversity_grid = np.full((len(turns), len(agents)), np.nan)
    rewards_grid = np.full((len(turns), len(agents)), 0.0)

    quartet_events = []  # (turn, agent, animal)

    for rec in data:
        t = int(rec["turn"])
        if t not in turn_to_row:
            continue
        r = turn_to_row[t]
        a = rec["agent"]
        c = agent_idx[a]
        progress_grid[r, c] = rec.get("progress", 0.0)
        diversity_grid[r, c] = rec.get("diversity", 0.0)
        rewards_grid[r, c] += rec.get("reward", 0.0)

        # detect quartet completion events by checking 'quartets' field
        qs = rec.get("quartets", [])
        if qs:
            for q in qs:
                quartet_events.append((t, a, q))

    # Plot progress per agent
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for a, agent in enumerate(agents):
        axes[0].plot(turns, progress_grid[:, a], label=agent, marker='o')
    axes[0].set_ylabel("Progress")
    axes[0].set_title("Per-turn Progress by Agent")
    axes[0].legend()
    axes[0].grid(True)

    # Mark quartet events
    for (t, agent, animal) in quartet_events:
        axes[0].axvline(x=t, color='gray', alpha=0.25)
        axes[0].text(t, axes[0].get_ylim()[1]*0.95, f"{agent}:{animal}", rotation=90, va='top', fontsize=8)

    # Diversity
    for a, agent in enumerate(agents):
        axes[1].plot(turns, diversity_grid[:, a], label=agent, marker='.', linestyle='--')
    axes[1].set_ylabel("Diversity")
    axes[1].set_title("Per-turn Diversity by Agent")
    axes[1].legend()
    axes[1].grid(True)

    # Rewards (stacked)
    bottoms = np.zeros(len(turns))
    for a, agent in enumerate(agents):
        axes[2].bar(turns, rewards_grid[:, a], bottom=bottoms, label=agent)
        bottoms += rewards_grid[:, a]
    axes[2].set_ylabel("Reward per Turn (stacked)")
    axes[2].set_xlabel("Turn")
    axes[2].set_title("Per-turn Rewards")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    out_png = Path(replay_path).with_suffix(".progress.png")
    plt.savefig(out_png, dpi=150)
    print(f"[SAVED] Progress visualization: {out_png}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_replay.py replays/game_YYYYMMDD_HHMMSS.jsonl")
        sys.exit(1)
    visualize(sys.argv[1])