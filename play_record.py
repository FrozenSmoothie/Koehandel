"""
Run a single game (or a few) and record a JSONL replay plus CSV summary.

Usage:
  python play_record.py --max_turns 300
  python play_record.py --checkpoint /path/to/checkpoint

Outputs:
  ./replays/game_YYYYMMDD_HHMMSS.jsonl
  ./replays/game_YYYYMMDD_HHMMSS_summary.csv
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from ray.rllib.algorithms.ppo import PPO
except Exception:
    PPO = None

from koehandel_game_engine import KoehandelPettingZooEnv


def sample_action_from_mask(mask):
    valid = np.nonzero(mask)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def sanitize(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:
        return None


def main(checkpoint=None, render=True, max_turns=None):
    Path("replays").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    replay_path = Path(f"replays/game_{ts}.jsonl")
    summary_path = Path(f"replays/game_{ts}_summary.csv")

    env = KoehandelPettingZooEnv(num_players=4, render_mode="human", max_turns=max_turns)
    env.reset()

    algo = None
    shared_policy_id = None
    if checkpoint and PPO is not None:
        try:
            algo = PPO.from_checkpoint(checkpoint)
            shared_policy_id = list(env.possible_agents)[0]
            print(f"[INFO] Loaded checkpoint {checkpoint}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")
            algo = None

    replay_file = replay_path.open("w", encoding="utf-8")
    summary_lines = ["turn,agent,action,reward,deck_size,phase,progress,diversity,quartets,trade_options\n"]

    step_index = 0
    try:
        for agent in env.agent_iter(max_iter=1000000):
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                if algo is not None:
                    try:
                        out = algo.compute_single_action(obs, policy_id=shared_policy_id, explore=False)
                        if isinstance(out, tuple):
                            action = out[0]
                        else:
                            action = out
                        if action is None:
                            if info and "action_mask" in info:
                                action = sample_action_from_mask(np.array(info["action_mask"], dtype=np.int8))
                            else:
                                action = int(env.action_space(agent).sample())
                    except Exception:
                        if info and "action_mask" in info:
                            action = sample_action_from_mask(np.array(info["action_mask"], dtype=np.int8))
                        else:
                            action = int(env.action_space(agent).sample())
                else:
                    if info and "action_mask" in info:
                        action = sample_action_from_mask(np.array(info["action_mask"], dtype=np.int8))
                    else:
                        action = int(env.action_space(agent).sample())

            env.step(action)
            step_index += 1

            current_turn = env.game.turn
            deck_size = len(env.game.animal_deck)
            phase = env.game.phase
            player = env.game.players[agent]
            progress = float(player.get_progress_score(env.game.quartet_values))
            diversity = float(player.get_diversity_score())
            quartets = [a for a, c in player.get_animal_counts().items() if c == 4]

            rec = {
                "global_step": int(step_index),
                "turn": int(current_turn),
                "agent": str(agent),
                "action": None if action is None else int(action),
                "reward": float(reward) if reward is not None else 0.0,
                "termination": bool(termination),
                "truncation": bool(truncation),
                "deck_size": int(deck_size),
                "phase": str(phase),
                "progress": float(progress),
                "diversity": float(diversity),
                "quartets": quartets,
                "info": sanitize(info),
                "timestamp": float(time.time()),
            }
            safe_rec = sanitize(rec)
            replay_file.write(json.dumps(safe_rec, ensure_ascii=False) + "\n")

            # add trade options summary safe string
            trade_opt = ""
            if info and isinstance(info, dict):
                to = info.get("trade_options")
                if to:
                    # fixed f-string syntax (no backslashes)
                    trade_opt = "|".join([f"{t['target']}:{t['animal']}:{t['mode']}@{t['action']}" for t in to])

            summary_lines.append(f"{current_turn},{agent},{safe_rec['action']},{safe_rec['reward']},{deck_size},{phase},{progress:.3f},{diversity:.3f},\"{quartets}\",\"{trade_opt}\"\n")

            if all(env.terminations.values()) or all(env.truncations.values()):
                print("[INFO] Episode finished")
                break

            if env.game.is_game_over():
                print("[INFO] Game over reached")
                break

    finally:
        replay_file.close()
        with summary_path.open("w", encoding="utf-8") as f:
            f.writelines(summary_lines)
        print(f"[SAVED] Replay: {replay_path}")
        print(f"[SAVED] Summary: {summary_path}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_turns", type=int, default=300)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, render=not args.no_render, max_turns=args.max_turns)