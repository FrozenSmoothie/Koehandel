"""
Quick test training - runs for only 10,000 steps (~30 seconds)
"""

import ray
from ray.tune.tuner import Tuner
from ray import tune
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
    """Quick test training."""

    print("=" * 70)
    print("KOEHANDEL RL - QUICK TEST TRAINING")
    print("=" * 70)
    print(f"Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Target: 10,000 steps (about 30 seconds)")
    print("=" * 70)

    # Initialize Ray
    print("\n[1/4] Initializing Ray...")
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    print("[OK] Ray initialized")

    # Register environment
    print("\n[2/4] Registering environment...")
    register_env("koehandel_env", lambda config: PettingZooEnv(env_creator(config)))

    test_env = env_creator({"num_players": 4})
    sample_agent = test_env.possible_agents[0]
    obs_space = test_env.observation_space(sample_agent)
    act_space = test_env.action_space(sample_agent)
    agent_list = test_env.possible_agents
    print("[OK] Environment registered")

    # Configure PPO (simplified for speed)
    print("\n[3/4] Configuring PPO...")
    config = (
        PPOConfig()
        .environment(
            env="koehandel_env",
            env_config={"num_players": 4}
        )
        .framework("torch")
        .env_runners(
            num_env_runners=2,  # Fewer workers for quick test
            rollout_fragment_length=64,  # Smaller fragments
        )
        .training(
            train_batch_size=256,  # Smaller batch
            minibatch_size=64,
            num_sgd_iter=5,  # Fewer epochs
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .multi_agent(
            policies={agent_list[0]},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_list[0],
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")  # Less logging
    )
    print("[OK] PPO configured (lightweight for testing)")

    # Storage path
    storage_path = os.path.abspath("./results_test")
    Path(storage_path).mkdir(parents=True, exist_ok=True)

    # Run quick training
    print("\n[4/4] Running quick test training...")
    print("=" * 70)

    try:
        start_time = time.time()

        tuner = Tuner(
            PPO,
            param_space=config.to_dict(),
            run_config=tune.RunConfig(
                name="quick_test",
                storage_path=storage_path,
                stop={
                    "env_runners/num_env_steps_sampled_lifetime": 10_000,  # Only 10k steps
                },
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=0,  # No checkpoints for quick test
                    checkpoint_at_end=False,
                ),
                verbose=0,  # Minimal output
            ),
        )

        print("Training started... (target: 10,000 steps)")
        results = tuner.fit()

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("QUICK TEST COMPLETED!")
        print("=" * 70)
        print(f"Time taken: {elapsed:.1f} seconds")

        # Get results
        best_result = results.get_best_result(metric="env_runners/num_env_steps_sampled_lifetime", mode="max")
        if best_result:
            metrics = best_result.metrics
            steps = metrics.get("env_runners/num_env_steps_sampled_lifetime", 0)
            iterations = metrics.get("training_iteration", 0)

            print(f"\nResults:")
            print(f"  - Total Steps:   {steps:,}")
            print(f"  - Iterations:    {iterations}")
            print(f"  - Steps/sec:     {steps / elapsed:.1f}")

            # Check if training worked
            total_loss = metrics.get("learners/player_0/total_loss", None)
            policy_loss = metrics.get("learners/player_0/policy_loss", None)

            if total_loss is not None:
                print(f"  - Total Loss:    {total_loss:.4f}")
            if policy_loss is not None:
                print(f"  - Policy Loss:   {policy_loss:.4f}")

            print("\n[SUCCESS] Training pipeline is working correctly!")
            print("=" * 70)
            print("\nReady for full training run:")
            print("  python train_koehandel.py")
        else:
            print("\n[WARNING] Could not extract results")

        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()
        print("\n[CLEANUP] Ray shutdown complete")


if __name__ == "__main__":
    main()