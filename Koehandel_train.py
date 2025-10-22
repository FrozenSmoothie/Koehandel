"""
Train script for Koehandel (Linux/Windows-friendly, Ray-fallback, small model).

This patched file contains compatibility fallbacks for a range of Ray/RLlib/Tune
versions. Notable compatibility measures included here:
- Safe Ray init with extended startup wait.
- Compatibility helper for PPOConfig env/rollout APIs (.env_runners vs .rollouts).
- Compatibility helper for setting model config (.model callable vs dict/attr).
- Disables RLlib "new API stack" by default so legacy .model() chaining works.
- Defensive TuneConfig construction (only pass resources_per_trial when supported).

Auto-warmup has been removed per request.
"""
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from inspect import signature

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
PARAMETERS: Dict[str, Any] = {
    # CPU budget: None -> auto-detect (reserve one)
    "use_cpus": None,
    "debug_driver_sampling": False,
    "num_env_runners": None,
    # Full training hyperparams
    "train_batch_size_full": 1024,
    "minibatch_size_full": 128,
    "rollout_fragment_length_full": 256,
    "num_epochs_full": 3,
    "lr_full": 5e-5,
    # Fast/debug presets
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
    # Stop criteria
    "stop_steps_debug": 10_000,
    "stop_steps_fast": 5_000,
    "stop_steps_short_full": 200_000,
    "stop_steps_long_full": 1_000_000,
    # Auto-warmup removed
    "auto_warmup": False,
    "warmup_seconds": 60,
    # Model size
    "model_fcnet_hiddens": [64, 64],
    "max_env_runners_cap": 16,
}

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("KOEHANDEL_RESULTS", BASE_DIR / "results"))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Helpers (compat wrappers)
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
    """
    os.environ.setdefault("RAY_raylet_start_wait_time_s", "300")
    try:
        ray.shutdown()
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=num_cpus)
        return True
    except Exception as e:
        print(f"[WARN] ray.init failed: {e}")
        try:
            ray.shutdown()
        except Exception:
            pass
        return False


def _apply_env_runners_compat(cfg_obj, num_env_runners: int, rollout_fragment_length, sample_timeout_s: float = 300.0):
    """
    Try to set env/rollout options in a way compatible across RLlib versions.
    rollout_fragment_length may be int or the string 'auto' (if supported).
    Returns the (possibly modified) cfg_obj.
    """
    # try .env_runners API first (some versions)
    try:
        return cfg_obj.env_runners(num_env_runners=num_env_runners, rollout_fragment_length=rollout_fragment_length, sample_timeout_s=sample_timeout_s)
    except Exception:
        pass
    # try .rollouts API (other versions)
    try:
        # new API uses num_rollout_workers
        return cfg_obj.rollouts(num_rollout_workers=num_env_runners, rollout_fragment_length=rollout_fragment_length)
    except Exception:
        pass
    # fallback: return unchanged
    return cfg_obj


def _set_model_compat(cfg_obj, model_dict: Dict[str, Any]):
    """
    Set model configuration on a PPOConfig-like object in a compatibility-safe way.
    Returns either the same object (PPOConfig) or a plain dict fallback (cfg_dict).
    """
    # Prefer disabling new API stack at the config creation site (we do that when building PPOConfig).
    # Try to call chainable .model(...) first
    try:
        maybe = getattr(cfg_obj, "model", None)
        if callable(maybe):
            return cfg_obj.model(model_dict)
    except TypeError:
        # .model exists but is a dict (non-callable)
        pass
    except Exception:
        pass

    # try to set as attribute (cfg.model = {...})
    try:
        setattr(cfg_obj, "model", model_dict)
        return cfg_obj
    except Exception:
        pass

    # last resort: convert to dict and inject model (Tuner accepts param_space={"config": <dict>})
    try:
        cfg_dict = cfg_obj.to_dict()
        cfg_dict["model"] = model_dict
        return cfg_dict
    except Exception:
        return cfg_obj


# ---------------------------
# Main
# ---------------------------
def main(
    debug_driver_sampling: Optional[bool] = None,
    fast: bool = False,
    long: bool = False,
    override_use_cpus: Optional[int] = None,
    skip_warmup_flag: bool = False,  # kept for CLI compatibility but auto-warmup is disabled
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

    # Try Ray; fallback to debug in-driver sampling if initialization fails
    ray_ok = safe_ray_init(use_cpus)
    if not ray_ok:
        print("[WARN] Ray failed to initialize; falling back to debug in-driver sampling (single-process).")
        debug_driver_sampling = True

    # Register environment
    register_env("koehandel_env", lambda c: PettingZooEnv(env_creator(c)))

    # Create a sample env to fetch spaces (helps build policies)
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
            train_batch_size = cfg["train_batch_size_full"]
            minibatch_size = cfg["minibatch_size_full"]
            num_epochs = cfg["num_epochs_full"]
            lr = cfg["lr_full"]
            # compute a sensible rollout_fragment_length so RLlib validations pass:
            candidate = max(1, int(train_batch_size // max(1, num_env_runners)))
            if candidate * num_env_runners == train_batch_size:
                rollout_fragment_length = candidate
            else:
                rollout_fragment_length = "auto"
            stop_criteria = {"num_env_steps_sampled_lifetime": cfg["stop_steps_short_full"] if not long else cfg["stop_steps_long_full"]}

    print(f"[CONFIG] num_env_runners={num_env_runners}, rollout_fragment_length={rollout_fragment_length}")
    print(f"[CONFIG] train_batch_size={train_batch_size}, minibatch_size={minibatch_size}, num_epochs={num_epochs}, lr={lr}")
    print(f"[CONFIG] stop_criteria={stop_criteria}")

    # Build PPO config (compat-friendly). Disable the new API stack to keep legacy chaining semantics.
    base_cfg = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="koehandel_env", env_config={"num_players": 4, "max_turns": 250}, disable_env_checking=True)
        .framework("torch")
    )
    base_cfg = _apply_env_runners_compat(base_cfg, num_env_runners=num_env_runners, rollout_fragment_length=rollout_fragment_length, sample_timeout_s=300.0)
    base_cfg = base_cfg.training(
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

    cfg_or_dict = _set_model_compat(base_cfg, {"fcnet_hiddens": PARAMETERS.get("model_fcnet_hiddens", [64, 64])})

    # multi_agent and resources (try chain-safe)
    try:
        if isinstance(cfg_or_dict, dict):
            # inject multi-agent into dict form
            cfg_or_dict.setdefault("multiagent", {})
            cfg_or_dict["multiagent"]["policies"] = {"shared_policy": (None, obs_space, act_space, {})}
            cfg_or_dict["multiagent"]["policy_mapping_fn"] = (lambda aid, ep, **k: "shared_policy")
            cfg_or_dict["num_gpus"] = 0
        else:
            cfg_or_dict = cfg_or_dict.multi_agent(policies={"shared_policy": (None, obs_space, act_space, {})}, policy_mapping_fn=lambda aid, ep, **k: "shared_policy")
            cfg_or_dict = cfg_or_dict.resources(num_gpus=0)
            cfg_or_dict = cfg_or_dict.debugging(log_level="INFO")
    except Exception:
        pass

    storage_path = str(RESULTS_ROOT.resolve())
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Results saver: {storage_path}/koehandel_training/")

    # Prepare Tune param_space depending on cfg_or_dict type
    from ray import tune as _tune
    if isinstance(cfg_or_dict, dict):
        param_space = {"config": cfg_or_dict}
    else:
        try:
            param_space = cfg_or_dict.to_dict()
        except Exception:
            param_space = {"config": cfg_or_dict}

    # Build TuneConfig defensively (only pass resources_per_trial if supported)
    tune_kwargs = {"num_samples": 1}
    try:
        params = signature(_tune.TuneConfig.__init__).parameters
        if "resources_per_trial" in params:
            tune_kwargs["resources_per_trial"] = {"cpu": use_cpus, "gpu": 0}
    except Exception:
        pass
    tune_conf = _tune.TuneConfig(**tune_kwargs)

    tuner = Tuner(
        PPO,
        param_space=param_space,
        tune_config=tune_conf,
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
            best_key = list(stop_criteria.keys())[0] if stop_criteria else None
            if best_key:
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
    # keep --no-warmup for CLI compatibility though auto-warmup is disabled by default
    parser.add_argument("--no-warmup", action="store_true", help="No-op (auto-warmup disabled).")
    args = parser.parse_args()

    main(debug_driver_sampling=None if not args.debug else True, fast=args.fast, long=args.long, override_use_cpus=args.cpus, skip_warmup_flag=args.no_warmup)