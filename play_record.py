"""
Run one game and record a detailed replay for visualization.

Usage:
  # Run with random (masked) actions (debug sampling in driver):
  python play_and_record.py

  # Run using a saved PPO checkpoint (single shared policy):
  python play_and_record.py --checkpoint /path/to/checkpoint

Outputs:
  ./replays/game_YYYYMMDD_HHMMSS.jsonl   (one JSON line per step)
  ./replays/game_YYYYMMDD_HHMMSS_summary.csv
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# RLlib imports (optional - only used if checkpoint provided)
try:
    from ray.rllib.algorithms.ppo import PPO
except Exception:
    PPO = None

from koehandel_game_engine import KoehandelPettingZooEnv


def sample_action_from_mask(mask):
    """Pick a random valid action from a binary action mask (1=valid)."""
    valid_idxs = np.nonzero(mask)[0]
    if len(valid_idxs) == 0:
        return 0
    return int(np.random.choice(valid_idxs))


def main(checkpoint=None, render=True, max_turns=None):
    Path("replays").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = Path(f"replays/game_{timestamp}.jsonl")
    summary_path = Path(f"replays/game_{timestamp}_summary.csv")

    # Create env
    env = KoehandelPettingZooEnv(num_players=4, render_mode="human", max_turns=max_turns or 300)
    env.reset()

    # Load policy if checkpoint provided
    algo = None
    shared_policy_id = None
    if checkpoint and PPO is not None:
        try:
            print(f"[INFO] Loading checkpoint: {checkpoint}")
            algo = PPO.from_checkpoint(checkpoint)
            # We used a shared policy ID of the form 'player_0' during training config
            shared_policy_id = list(env.possible_agents)[0]
            print("[INFO] Checkpoint loaded.")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}. Falling back to random policy.")
            algo = None

    step_index = 0
    replay_file = replay_path.open("w", encoding="utf-8")
    summary_lines = []
    # CSV header
    summary_lines.append("turn,agent,action,reward,deck_size,phase,progress,diversity,quartets\n")

    try:
        # agent_iter style stepping
        for agent in env.agent_iter(max_iter=10_000_000):
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                # Prefer the trained policy (single shared) if available
                if algo is not None:
                    # compute_single_action uses the algorithm's inference.
                    try:
                        # Provide action_mask (if present) as part of kwargs so policy sees it
                        mask = info.get("action_mask", None)
                        if mask is not None:
                            action, _, _ = algo.compute_single_action(obs, policy_id=shared_policy_id, explore=False, input_dict=None, full_pass=False)
                        else:
                            action, _, _ = algo.compute_single_action(obs, policy_id=shared_policy_id, explore=False)
                        # If algo returned None or invalid, fallback to mask sampling
                        if action is None:
                            if info.get("action_mask") is not None:
                                action = int(sample_action_from_mask(np.array(info["action_mask"], dtype=np.int8)))
                            else:
                                action = int(env.action_space(agent).sample())
                    except Exception:
                        # If compute_single_action fails (multi-agent mismatch), fallback
                        if info.get("action_mask") is not None:
                            action = int(sample_action_from_mask(np.array(info["action_mask"], dtype=np.int8)))
                        else:
                            action = int(env.action_space(agent).sample())
                else:
                    # Random but valid action using action_mask when available
                    if info.get("action_mask") is not None:
                        action = int(sample_action_from_mask(np.array(info["action_mask"], dtype=np.int8)))
                    else:
                        action = int(env.action_space(agent).sample())

            # Step env
            env.step(action)
            step_index += 1

            # Capture state for replay
            current_turn = env.game.turn
            deck_size = len(env.game.animal_deck)
            phase = env.game.phase
            player = env.game.players[agent]
            progress = float(player.get_progress_score(env.game.quartet_values))
            diversity = float(player.get_diversity_score())
            quartets = list(a for a, c in player.get_animal_counts().items() if c == 4)

            record = {
                "global_step": step_index,
                "turn": current_turn,
                "agent": agent,
                "action": None if action is None else int(action),
                "reward": float(reward) if reward is not None else 0.0,
                "termination": bool(termination),
                "truncation": bool(truncation),
                "deck_size": deck_size,
                "phase": phase,
                "progress": progress,
                "diversity": diversity,
                "quartets": quartets,
                "info": info,
                "timestamp": time.time(),
            }
            replay_file.write(json.dumps(record) + "\n")

            # Append CSV summary line
            summary_lines.append(f"{current_turn},{agent},{record['action']},{record['reward']},{deck_size},{phase},{progress:.3f},{diversity:.3f},\"{quartets}\"\n")

            # Print rendering already printed by env.render() when render_mode='human'
            # If you want extra console output per step, uncomment:
            # print(f"[STEP {step_index}] turn={current_turn} agent={agent} action={action} reward={reward}")

            # Stop if all agents are terminated/truncated
            if all(env.terminations.values()) or all(env.truncations.values()):
                print("[INFO] All agents terminated/truncated - episode finished.")
                break

            # Safety: stop if deck empty and phase is game_over
            if env.game.is_game_over():
                print("[INFO] Game over reached.")
                break

    finally:
        replay_file.close()
        # Write summary CSV
        with summary_path.open("w", encoding="utf-8") as f:
            f.writelines(summary_lines)
        print(f"[SAVED] Replay JSONL: {replay_path}")
        print(f"[SAVED] Summary CSV: {summary_path}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO checkpoint (optional)")
    parser.add_argument("--max_turns", type=int, default=300, help="Max turns (env truncation)")
    parser.add_argument("--no-render", action="store_true", help="Do not call env.render()")
    args = parser.parse_args()

    main(checkpoint=args.checkpoint, render=not args.no_render, max_turns=args.max_turns)