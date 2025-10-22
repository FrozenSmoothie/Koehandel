"""
Train script for Koehandel (Linux-friendly, Ray-fallback, auto-warmup, small model).

Features in this patched file:
- Top-level editable PARAMETERS for quick tuning.
- Uses KOEHANDEL_RESULTS env var (set by systemd run wrapper) or ./results next to the script.
- Safe Ray init with extended startup wait and automatic fallback to in-driver sampling on failures.
- Optional automatic warmup profiler to measure steps/sec and auto-set stop steps for a ~30-minute run.
- Requests the cluster CPUs for the Tune trial so trials actually receive the configured CPU budget.
- Uses a compact policy network (.model({"fcnet_hiddens": [64, 64]})) to speed up training on CPU.
- Works on Linux and Windows (path handling via pathlib).

Usage:
  python train_koehandel.py            # runs with PARAMETERS defaults
  python train_koehandel.py --fast     # quick smoke run
  python train_koehandel.py --debug    # force in-driver sampling
  python train_koehandel.py --long     # target long (1M) run
  python train_koehandel.py --no-warmup # skip auto warmup profiling if enabled in PARAMETERS
"""
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import ray
from ray import tune
from ray.tune.tuner import Tuner
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

from koehandel_game_engine import KoehandelPettingZooEnv

# ---------------------------
# TOP-LEVEL PARAMETERS
# ---------------------------
# Edit these to adjust default behavior on the server.
PARAMETERS: Dict[str, Any] = {
    # CPU budget:
    # - None: auto-detect (os.cpu_count() - 1)
    # - Integer: use this many CPU slots for the trial (recommended: reserve 1 for OS)
    "use_cpus": None,  # set to 3 on a 4-core Hetzner (reserves 1), or None to auto-detect

    # Sampling mode default:
    # - True -> debug/in-driver sampling (single-process)
    # - False -> full parallel sampling
    "debug_driver_sampling": False,

    # Number of parallel env runners / rollout workers:
    # - None: computed from use_cpus (use_cpus - 1, capped)
    "num_env_runners": None,

    # ---------------- Full training hyperparameters ----------------
    # Smaller defaults for faster iterations; adjust if you want more stable training.
    "train_batch_size_full": 1024,
    "minibatch_size_full": 128,
    "rollout_fragment_length_full": 256,
    "num_epochs_full": 3,
    "lr_full": 5e-5,

    # ---------------- Fast/debug presets ----------------
    "train_batch_size_fast": 256,
    "minibatch_size_fast": 64,
    "rollout_fragment_length_fast": 16,
    "num_epochs_fast": 2,
    "lr_fast": 3e-4,

    "train_batch_size_debug": 128,
    "minibatch_size_debug": 64,
    "rollout_fragment_length_debug": 32,
    "num_epochs_debug": 2,
    "lr_debug": 1e-4,

    # ---------------- Stop criteria (environment steps) ----------------
    "stop_steps_debug": 10_000,
    "stop_steps_fast": 5_000,
    "stop_steps_short_full": 200_000,
    "stop_steps_long_full": 1_000_000,

    # ---------------- Auto-warmup ----------------
    # If True, perform a short warmup run to estimate steps/sec and auto-set
    # stop_steps_short_full so the following full run targets ~30 minutes.
    # Set to False if you prefer to keep stop_steps_short_full unchanged.
    "auto_warmup": True,
    "warmup_seconds": 60,  # length of warmup profiling (seconds)

    # Model size: smaller networks run much faster on CPU.
    "model_fcnet_hiddens": [64, 64],

    # Cap for auto env runners
    "max_env_runners_cap": 16,
}

# ---------------------------
# Paths: use environment-provided results path when available
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("KOEHANDEL_RESULTS", BASE_DIR / "results"))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Helpers
# ---------------------------
def _auto_detect_cpus(reserve_one: bool = True) -> int:
    cnt = os.cpu_count() or 1
    if reserve_one and cnt > 1:
        return max(1, cnt - 1)
    return cnt


def env_creator(config):
    return KoehandelPettingZooEnv(num_players=config.get("num_players", 4), max_turns=config.get("max_turns", None))


def safe_ray_init(num_cpus: int) -> bool:
    """
    Attempt to initialize Ray with a higher startup wait time.
    Returns True if Ray initialized successfully, False otherwise.
    On failure, prints the exception and returns False.
    """
    # Give ray more time to start its processes (helps on slower machines)
    os.environ.setdefault("RAY_raylet_start_wait_time_s", "300")
    try:
        ray.shutdown()  # ensure clean
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=num_cpus)
        return True
    except Exception as e:
        print(f"[WARN] ray.init failed: {e}")
        try:
            ray.shutdown()
        except Exception:
            pass
        return False


def measure_warmup_steps_per_sec(warmup_seconds: int, cfg_override: Dict[str, Any], use_cpus: int) -> float:
    """
    Run a short warmup by building a small PPO algorithm and calling train()
    repeatedly for warmup_seconds. Returns measured steps/sec.
    This avoids using Tune for the warmup (lighter weight).
    """
    print(f"[WARMUP] Running warmup profiling for {warmup_seconds}s to estimate steps/sec...")
    # Build minimal config for warmup (use debug/in-driver sampling or few workers)
    # Use PPOConfig but keep it small and fast
    warm_cfg = (
        PPOConfig()
        .environment(env="koehandel_env", env_config={"num_players": 4}, disable_env_checking=True)
        .framework("torch")
        .env_runners(num_env_runners=cfg_override.get("num_env_runners", 0),
                     rollout_fragment_length=cfg_override.get("rollout_fragment_length", 32),
                     sample_timeout_s=300.0)
        .training(
            train_batch_size=cfg_override.get("train_batch_size", 256),
            minibatch_size=cfg_override.get("minibatch_size", 64),
            num_epochs=cfg_override.get("num_epochs", 1),
            lr=cfg_override.get("lr", 1e-4),
        )
        .model({"fcnet_hiddens": PARAMETERS.get("model_fcnet_hiddens", [64, 64])})
        .multi_agent(policies={"shared_policy": (None, None, None, {})},
                     policy_mapping_fn=lambda aid, ep, **k: "shared_policy")
        .resources(num_gpus=0)
    )

    # Build and run
    algo = warm_cfg.build()
    start = time.time()
    collected_steps = 0
    try:
        while True:
            res = algo.train()
            # Try to extract steps_this_iter or cumulative timesteps
            steps_this_iter = res.get("timesteps_this_iter") or res.get("timesteps_total") or res.get("num_env_steps_sampled") or res.get("num_env_steps_sampled_lifetime") or 0
            # timesteps_this_iter expected to be >0; but safer: use timesteps_this_iter
            collected_steps += int(steps_this_iter or 0)
            elapsed = time.time() - start
            if elapsed >= warmup_seconds:
                break
    except KeyboardInterrupt:
        print("[WARMUP] interrupted by user")
    finally:
        algo.stop()

    if elapsed <= 0:
        return 0.0
    steps_per_sec = collected_steps / elapsed
    print(f"[WARMUP] Collected {collected_steps} steps in {elapsed:.1f}s -> {steps_per_sec:.2f} steps/sec")
    return max(0.0, steps_per_sec)


# ---------------------------
# Main
# ---------------------------
def main(
    debug_driver_sampling: Optional[bool] = None,
    fast: bool = False,
    long: bool = False,
    override_use_cpus: Optional[int] = None,
    skip_warmup_flag: bool = False,
):
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
    print("KOEHANDEL RL TRAINING (PATCHED)")
    print("=" * 70)
    print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"[CONFIG] debug_driver_sampling={debug_driver_sampling}, fast={fast}, long={long}, auto_warmup={cfg['auto_warmup']}")
    print(f"[CONFIG] use_cpus={use_cpus}")
    print("=" * 70)

    # Try to initialize Ray; if it fails, fallback to debug in-driver sampling.
    ray_ok = safe_ray_init(use_cpus)
    if not ray_ok:
        print("[WARN] Ray failed to initialize; falling back to debug in-driver sampling (single-process).")
        debug_driver_sampling = True
        # Note: do not raise; we continue running without ray worker processes.

    # Register environment
    register_env("koehandel_env", lambda c: PettingZooEnv(env_creator(c)))

    # Create a sample env to fetch spaces
    sample_env = env_creator({"num_players": 4})
    sample_agent = sample_env.possible_agents[0]
    obs_space = sample_env.observation_space(sample_agent)
    act_space = sample_env.action_space(sample_agent)
    try:
        sample_env.close()
    except Exception:
        pass

    # Prepare mode-specific settings
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
            print("[MODE] FULL - parallel sampling")
            if cfg["num_env_runners"] is not None:
                num_env_runners = cfg["num_env_runners"]
            else:
                num_env_runners = max(1, min(use_cpus - 1, cfg.get("max_env_runners_cap", 16)))
            rollout_fragment_length = cfg["rollout_fragment_length_full"]
            train_batch_size = cfg["train_batch_size_full"]
            minibatch_size = cfg["minibatch_size_full"]
            num_epochs = cfg["num_epochs_full"]
            lr = cfg["lr_full"]
            # decide stop_criteria; may be adjusted by auto-warmup below
            stop_criteria = {"num_env_steps_sampled_lifetime": cfg["stop_steps_short_full"] if not long else cfg["stop_steps_long_full"]}

    print(f"[CONFIG] num_env_runners={num_env_runners}, rollout_fragment_length={rollout_fragment_length}")
    print(f"[CONFIG] train_batch_size={train_batch_size}, minibatch_size={minibatch_size}, num_epochs={num_epochs}, lr={lr}")
    print(f"[CONFIG] stop_criteria={stop_criteria}")

    # If auto-warmup is enabled and we're in full mode, run a short profiler to auto-set stop steps.
    if cfg.get("auto_warmup", False) and not fast and not debug_driver_sampling and not skip_warmup_flag:
        try:
            warm_cfg_override = {
                "num_env_runners": 0,  # warmup: in-driver to measure single-process steps/sec reliably
                "rollout_fragment_length": cfg.get("rollout_fragment_length_debug", 32),
                "train_batch_size": cfg.get("train_batch_size_debug", 128),
                "minibatch_size": cfg.get("minibatch_size_debug", 64),
                "num_epochs": cfg.get("num_epochs_debug", 1),
                "lr": cfg.get("lr_debug", 1e-4),
            }
            steps_sec = measure_warmup_steps_per_sec(cfg.get("warmup_seconds", 60), warm_cfg_override, use_cpus)
            if steps_sec > 1.0:
                desired_seconds = 30 * 60  # 30 minutes
                computed_stop = int(steps_sec * desired_seconds)
                # set a sensible lower/upper bound
                computed_stop = max(50_000, min(computed_stop, cfg.get("stop_steps_long_full", 1_000_000)))
                stop_criteria = {"num_env_steps_sampled_lifetime": computed_stop}
                print(f"[AUTO-WARMUP] Auto-set stop_criteria to {stop_criteria} based on {steps_sec:.2f} steps/sec")
            else:
                print("[AUTO-WARMUP] Warmup measured too low steps/sec; skipping auto-stop adjustment.")
        except Exception as e:
            print(f"[AUTO-WARMUP] Warmup failed: {e}")

    # Build PPO config
    config = (
        PPOConfig()
        .environment(env="koehandel_env", env_config={"num_players": 4, "max_turns": 250}, disable_env_checking=True)
        .framework("torch")
        .env_runners(num_env_runners=num_env_runners, rollout_fragment_length=rollout_fragment_length, sample_timeout_s=300.0)
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
        .model({"fcnet_hiddens": PARAMETERS.get("model_fcnet_hiddens", [64, 64])})
        .multi_agent(policies={"shared_policy": (None, obs_space, act_space, {})}, policy_mapping_fn=lambda aid, ep, **k: "shared_policy")
        .resources(num_gpus=0)
        .debugging(log_level="INFO")
    )

    storage_path = str(RESULTS_ROOT.resolve())
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Results saver: {storage_path}/koehandel_training/")

    # Configure Tune to request cluster resources per trial (so the trial actually gets the CPUs)
    from ray import tune as _tune

    tuner = Tuner(
        PPO,
        param_space=config.to_dict(),
        tune_config=_tune.TuneConfig(resources_per_trial={"cpu": use_cpus, "gpu": 0}, num_samples=1),
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
            # Attempt to print a few useful metrics if available
            best_key = list(stop_criteria.keys())[0]
            best = results.get_best_result(metric=best_key, mode="max")
            if best:
                metrics = best.metrics
                for k in ("num_env_steps_sampled_lifetime", "training_iteration", "episode_reward_mean", "timesteps_total"):
                    if k in metrics:
                        print(f"  - {k}: {metrics[k]}")
        except Exception:
            pass
        print(f"[OK] Training run saved to: {storage_path}/koehandel_training/")
    except KeyboardInterrupt:
        print("[WARN] Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
        print("[OK] Ray shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Koehandel with configurable PARAMETERS.")
    parser.add_argument("--debug", action="store_true", help="Run debug in-driver sampling (single-threaded).")
    parser.add_argument("--fast", action="store_true", help="Run a very short fast test (overrides debug).")
    parser.add_argument("--long", action="store_true", help="Run a full-length training (stop at 1M steps).")
    parser.add_argument("--cpus", type=int, default=None, help="Override CPU budget (num_cpus for Ray).")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the auto warmup profiler even if enabled in PARAMETERS.")
    args = parser.parse_args()

    main(debug_driver_sampling=None if not args.debug else True, fast=args.fast, long=args.long, override_use_cpus=args.cpus, skip_warmup_flag=args.no_warmup)