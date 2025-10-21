"""
Diagnose what's happening with actions in the environment
"""

from koehandel_game_engine import KoehandelPettingZooEnv
import numpy as np


def test_environment():
    """Test if environment works at all."""
    print("=" * 70)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)

    env = KoehandelPettingZooEnv(num_players=4)
    env.reset(seed=42)

    print("\n[1] Testing basic functionality...")

    step_count = 0
    valid_action_counts = []
    rewards_received = []

    for agent in env.agent_iter(max_iter=100):
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"  Agent {agent} terminated at step {step_count}")
            action = None
        else:
            # Get valid actions
            valid_actions = env.game.get_valid_actions(agent)
            valid_action_counts.append(len(valid_actions))

            if step_count < 10:
                print(f"\n  Step {step_count} - Agent: {agent}")
                print(f"    Valid actions: {len(valid_actions)} options")
                print(f"    Sample actions: {valid_actions[:10]}")
                print(f"    Observation shape: {observation.shape}")
                print(f"    Reward: {reward:.4f}")

            # Take random valid action
            if valid_actions:
                action = np.random.choice(valid_actions)
            else:
                action = 0

            rewards_received.append(reward)

        env.step(action)
        step_count += 1

        if step_count >= 100:
            break

    print(f"\n[2] Summary after {step_count} steps:")
    print(f"  Avg valid actions per step: {np.mean(valid_action_counts):.1f}")
    print(f"  Avg reward per step: {np.mean(rewards_received):.6f}")
    print(f"  Total reward sum: {np.sum(rewards_received):.6f}")
    print(f"  Non-zero rewards: {np.sum(np.array(rewards_received) != 0)}/{len(rewards_received)}")

    # Check if game ended naturally
    if env.game.is_game_over():
        print(f"\n[3] Game ended naturally")
        for agent_name, player in env.game.players.items():
            animals = player.get_animal_counts()
            quartets = [a for a, c in animals.items() if c == 4]
            print(f"  {agent_name}: {quartets} quartets, score={player.score}")
    else:
        print(f"\n[3] Game did not finish (still in {env.game.phase} phase)")
        print(f"  Cards remaining in deck: {len(env.game.animal_deck)}")

    print("=" * 70)


if __name__ == "__main__":
    test_environment()