"""
Train script for Koehandel with top-of-file adjustable parameters.

This variant reduces batch sizes, epochs, and run-length to avoid very long runs
(aims for ~30 minutes or less depending on your hardware). Edit the PARAMETERS
block to tweak further.

Usage:
  python train_koehandel.py            # uses PARAMETERS defaults
  python train_koehandel.py --fast     # very short smoke test
  python train_koehandel.py --debug    # in-driver sampling
  python train_koehandel.py --long     # full 1M-step run (overrides short stop)
"""
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import ray
from ray import tune
from ray.tune.tuner import Tuner
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from koehandel_game_engine import KoehandelPettingZooEnv

# ===============================================================
# === TOP-LEVEL PARAMETERS: edit these to change default runs ===
# ===============================================================
# Lowered defaults to avoid extremely long runs by default.
# Change values if you want faster (less stable) or slower (more stable) runs.
PARAMETERS = {
    # CPU budget:
    # - None: auto-detect (os.cpu_count()-1)
    # - Integer: force Ray to use exactly this many CPU slots
    "use_cpus": 3,  # keep full CPU budget for throughput on your 12-core PC

    # Sampling mode default:
    # - True => debug (in-driver) sampling (single-process)
    # - False => full (parallel sampling)
    "debug_driver_sampling": False,

    # Number of parallel env runners / rollout workers:
    # Use fewer than the max to reduce contention and overhead.
    "num_env_runners": 2,  # use_cpus - 1 (driver + 2 workers) — reduces contention

    # ---------------- Full training hyperparameters (reduced for shorter runs) ---------------
    "train_batch_size_full": 4096,  # smaller than 8192 to shorten iteration time
    "minibatch_size_full": 256,  # divides train batch (4096 / 256 = 16 minibatches)
    "rollout_fragment_length_full": 256,  # larger fragments reduce IPC; not too large to avoid sample correlation
    "num_epochs_full": 2,  # fewer epochs per update to save compute
    "lr_full": 5e-5,  # keep your stable LR

    # ---------------- Fast test hyperparameters (very short runs) ---------------
    "train_batch_size_fast": 512,  # small fast test batches (keeps iterations short)
    "minibatch_size_fast": 64,
    "rollout_fragment_length_fast": 32,
    "num_epochs_fast": 2,
    "lr_fast": 3e-4,

    # ---------------- Debug (in-driver) mode hyperparameters ---------------
    "train_batch_size_debug": 128,
    "minibatch_size_debug": 64,
    "rollout_fragment_length_debug": 32,
    "num_epochs_debug": 2,
    "lr_debug": 1e-4,

    # ---------------- Stop criteria (when to stop training) ---------------
    "stop_steps_debug": 10_000,
    "stop_steps_fast": 5_000,
    "stop_steps_short_full": 100_000,  # lowered target for 4-CPU so a run finishes faster
    "stop_steps_long_full": 500_000
}
# ===============================================================
# End of top-level parameters
# ===============================================================


def env_creator(config):
    return KoehandelPettingZooEnv(num_players=config.get("num_players", 4), max_turns=config.get("max_turns", None))


def _auto_detect_cpus(reserve_one: bool = True) -> int:
    cnt = os.cpu_count() or 1
    if reserve_one and cnt > 1:
        return max(1, cnt - 1)
    return cnt


def main(
    debug_driver_sampling: Optional[bool] = None,
    fast: bool = False,
    long: bool = False,
    override_use_cpus: Optional[int] = None,
):
    """
    Main entry for training. Parameters passed from CLI override the top-level PARAMETERS.
    - debug_driver_sampling: if True, use in-driver sampling (single-process).
      If None, uses PARAMETERS["debug_driver_sampling"].
    - fast: run a very short quick test (overrides debug).
    - long: run a full long training (overrides debug flag to False).
    - override_use_cpus: override CPU budget.
    """
    cfg = PARAMETERS.copy()
    if debug_driver_sampling is None:
        debug_driver_sampling = cfg["debug_driver_sampling"]
    if fast:
        debug_driver_sampling = True
    if long:
        debug_driver_sampling = False

    use_cpus = override_use_cpus if override_use_cpus is not None else cfg["use_cpus"]
    if use_cpus is None:
        use_cpus = _auto_detect_cpus(reserve_one=True)

    print("=" * 70)
    print("KOEHANDEL RL TRAINING (CONFIGURABLE - SHORTER DEFAULTS)")
    print("=" * 70)
    print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"[CONFIG] debug_driver_sampling={debug_driver_sampling}, fast={fast}, long={long}")
    print(f"[CONFIG] use_cpus={use_cpus}")
    print("=" * 70)

    # Initialize Ray with the CPU budget
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=use_cpus)
    print("[OK] Ray initialized")

    # Register environment for RLlib
    register_env("koehandel_env", lambda cfg: PettingZooEnv(env_creator(cfg)))
    print("[OK] Environment registered as 'koehandel_env'")

    # Create a sample env to fetch spaces
    sample_env = env_creator({"num_players": 4})
    sample_agent = sample_env.possible_agents[0]
    obs_space = sample_env.observation_space(sample_agent)
    act_space = sample_env.action_space(sample_agent)
    agent_list = sample_env.possible_agents
    print(f"[OK] Sample environment: agents={agent_list}, obs_space={obs_space}, act_space={act_space}")
    try:
        sample_env.close()
    except Exception:
        pass

    # Choose settings based on mode
    if fast:
        print("[MODE] FAST - very short test")
        num_env_runners = 0
        rollout_fragment_length = cfg["rollout_fragment_length_fast"]
        train_batch_size = cfg["train_batch_size_fast"]
        minibatch_size = cfg["minibatch_size_fast"]
        num_epochs = cfg["num_epochs_fast"]
        lr = cfg["lr_fast"]
        stop_criteria = {"training_iteration": max(1, cfg["stop_steps_fast"] // (train_batch_size or 1))}
    else:
        if debug_driver_sampling:
            print("[MODE] DEBUG - in-driver sampling (single-process)")
            num_env_runners = 0
            rollout_fragment_length = cfg["rollout_fragment_length_debug"]
            train_batch_size = cfg["train_batch_size_debug"]
            minibatch_size = cfg["minibatch_size_debug"]
            num_epochs = cfg["num_epochs_debug"]
            lr = cfg["lr_debug"]
            stop_criteria = {"num_env_steps_sampled_lifetime": cfg["stop_steps_debug"]}
        else:
            print("[MODE] FULL - parallel sampling (shorter defaults)")
            if cfg["num_env_runners"] is not None:
                num_env_runners = cfg["num_env_runners"]
            else:
                num_env_runners = max(1, min(use_cpus - 1, 16))
            rollout_fragment_length = cfg["rollout_fragment_length_full"]
            train_batch_size = cfg["train_batch_size_full"]
            minibatch_size = cfg["minibatch_size_full"]
            num_epochs = cfg["num_epochs_full"]
            lr = cfg["lr_full"]
            stop_criteria = {"num_env_steps_sampled_lifetime": cfg["stop_steps_short_full"] if not long else cfg["stop_steps_long_full"]}

    print(f"[CONFIG] num_env_runners={num_env_runners}, rollout_fragment_length={rollout_fragment_length}")
    print(f"[CONFIG] train_batch_size={train_batch_size}, minibatch_size={minibatch_size}, num_epochs={num_epochs}, lr={lr}")
    print(f"[CONFIG] stop_criteria={stop_criteria}")

    # Build PPO config
    config = (
        PPOConfig()
        .environment(env="koehandel_env", env_config={"num_players": 4, "max_turns": 250}, disable_env_checking=True)
        .framework("torch")
        .env_runners(num_env_runners=num_env_runners, rollout_fragment_length=rollout_fragment_length, sample_timeout_s=300.0)
        # rely on env_runners(...) which is the modern API
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .multi_agent(policies={"shared_policy": (None, obs_space, act_space, {})}, policy_mapping_fn=lambda aid, ep, **k: "shared_policy")
        .resources(num_gpus=0)
        .debugging(log_level="INFO")
    )

    # Storage & tuner
    storage_path = os.path.abspath("./results")
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Results saver: {storage_path}/koehandel_training/")

    tuner = Tuner(
        PPO,
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            name="koehandel_training",
            storage_path=storage_path,
            stop=stop_criteria,
            checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True),
            verbose=1,
        ),
    )

    try:
        print("[INFO] Starting training...")
        start_time = time.time()
        results = tuner.fit()
        elapsed = time.time() - start_time
        print(f"[INFO] Training finished in {elapsed/60:.2f} minutes")
        try:
            best = results.get_best_result(metric=list(stop_criteria.keys())[0], mode="max")
            if best:
                metrics = best.metrics
                for k in ("num_env_steps_sampled_lifetime", "training_iteration", "episode_reward_mean"):
                    if k in metrics:
                        print(f"  - {k}: {metrics[k]}")
        except Exception:
            pass
        print(f"[OK] Training run saved to: {storage_path}/koehandel_training/")
    except KeyboardInterrupt:
        print("[WARN] Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] Training run failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()
        print("[OK] Ray shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Koehandel with configurable top-of-file PARAMETERS.")
    parser.add_argument("--debug", action="store_true", help="Run debug in-driver sampling (single-threaded).")
    parser.add_argument("--fast", action="store_true", help="Run a very short fast test (overrides debug).")
    parser.add_argument("--long", action="store_true", help="Run a full-length training (stop at 1M steps).")
    parser.add_argument("--cpus", type=int, default=None, help="Override CPU budget (num_cpus for Ray).")
    args = parser.parse_args()

    main(debug_driver_sampling=None if args.debug is False else True, fast=args.fast, long=args.long, override_use_cpus=args.cpus)