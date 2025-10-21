"""
Training script with OPTIMIZED settings for new API
"""

import ray
from ray import tune
from ray.tune.tuner import Tuner
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from koehandel_game_engine import KoehandelPettingZooEnv
import time
import os
from pathlib import Path
from datetime import datetime, timezone


def env_creator(config):
    """Create the PettingZoo wrapped environment."""
    env = KoehandelPettingZooEnv(num_players=config.get("num_players", 4))
    return env


def main():
    """Main training function with OPTIMIZED settings."""

    print("="*70)
    print("KOEHANDEL RL TRAINING (OPTIMIZED)")
    print("="*70)
    print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: FrozenSmoothie")
    print("="*70)

    # Initialize Ray
    print("\n[1/5] Initializing Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=6)  # Limit CPUs
    print("[OK] Ray initialized")

    # Register the environment
    print("\n[2/5] Registering environment...")
    register_env("koehandel_env", lambda config: PettingZooEnv(env_creator(config)))
    print("[OK] Environment registered")

    # Get sample environment
    print("\n[3/5] Creating sample environment...")
    test_env = env_creator({"num_players": 4})

    sample_agent = test_env.possible_agents[0]
    obs_space = test_env.observation_space(sample_agent)
    act_space = test_env.action_space(sample_agent)
    agent_list = test_env.possible_agents

    print(f"[OK] Sample environment created")
    print(f"     - Agents: {agent_list}")
    print(f"     - Observation space: {obs_space}")
    print(f"     - Action space: {act_space}")

    # Configure PPO with OPTIMIZED settings
    print("\n[4/5] Configuring PPO (OPTIMIZED FOR SLOW ENV)...")
    config = (
        PPOConfig()
        .environment(
            env="koehandel_env",
            env_config={"num_players": 4},
            disable_env_checking=True,  # Skip validation for speed
        )
        .framework("torch")
        .env_runners(
            num_env_runners=2,  # REDUCED from 4 (less overhead)
            rollout_fragment_length=64,  # REDUCED from 128 (faster returns)
            sample_timeout_s=180.0,  # INCREASED timeout (3 minutes)
        )
        .training(
            train_batch_size=256,  # REDUCED from 512 (faster training)
            minibatch_size=64,  # REDUCED from 128
            num_sgd_iter=5,  # REDUCED from 10 (faster iterations)
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={agent_list[0]},  # SINGLE SHARED POLICY
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_list[0],
        )
        .resources(
            num_gpus=0,
            num_cpus_for_main_process=1,
        )
        .debugging(
            log_level="ERROR",  # Less logging overhead
        )
    )
    print("[OK] PPO configured (OPTIMIZED)")
    print("     - 2 workers (reduced overhead)")
    print("     - Smaller batches (faster iterations)")
    print("     - 3 minute sample timeout")
    print("     - Single shared policy")

    # Set up storage
    storage_path = os.path.abspath("./results")
    Path(storage_path).mkdir(parents=True, exist_ok=True)

    print(f"\n     - Storage path: {storage_path}")
    print("\n[TIME ESTIMATE]")
    print("  With optimized settings: ~60-90 minutes for 1M steps")

    # Run training
    print("\n[5/5] Starting training...")
    print("="*70)
    print("[WAIT] First samples may take 1-2 minutes to collect...")
    print("="*70)

    try:
        tuner = Tuner(
            PPO,
            param_space=config.to_dict(),
            run_config=tune.RunConfig(
                name="koehandel_training",
                storage_path=storage_path,
                stop={
                    "num_env_steps_sampled_lifetime": 1_000_000,
                },
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=10,
                    checkpoint_at_end=True,
                ),
                verbose=1,
            ),
        )

        print("\n[INFO] Training started (be patient, first samples take time)")
        print("[INFO] Monitor with: python monitor_training.py")
        print()

        start_time = time.time()
        results = tuner.fit()

        print("\n" + "="*70)
        print("TRAINING COMPLETED!")
        print("="*70)

        elapsed = time.time() - start_time
        print(f"Total Time: {elapsed/60:.1f} minutes")

        best_result = results.get_best_result(metric="num_env_steps_sampled_lifetime", mode="max")
        if best_result:
            metrics = best_result.metrics
            print(f"\nFinal Results:")
            print(f"  - Env Steps: {metrics.get('num_env_steps_sampled_lifetime', 0):,.0f}")
            print(f"  - Iterations: {metrics.get('training_iteration', 0)}")

        print(f"\nResults saved to: {storage_path}/koehandel_training/")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED BY USER")
        print("="*70)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()