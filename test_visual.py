"""
Visual test - watch random agents play
"""

from koehandel_game_engine import KoehandelPettingZooEnv
import time
import random

def test_visual():
    """Watch random agents play with rendering."""
    env = KoehandelPettingZooEnv(num_players=4, render_mode="human")
    env.reset(seed=42)

    print("Watching random agents play Koehandel...")
    print("(Game will auto-play with random valid actions)\n")

    step_count = 0
    for agent in env.agent_iter(max_iter=500):
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Random action from valid actions
            valid_actions = env.game.get_valid_actions(agent)
            if valid_actions:
                # 70% chance to bid if possible, 30% to pass
                if len(valid_actions) > 1 and random.random() > 0.3:
                    action = random.choice(valid_actions[1:])  # Skip pass, choose a bid
                else:
                    action = 0  # Pass
            else:
                action = 0

        env.step(action)
        step_count += 1

        # Slow down for visibility
        time.sleep(0.3)

        if all(env.terminations.values()):
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            for agent_name in env.agents:
                player = env.game.players[agent_name]
                animals = player.get_animal_counts()
                winner = " [WINNER]" if env.rewards[agent_name] > 0 else ""

                # Count complete quartets
                quartets = [animal for animal, count in animals.items() if count == 4]

                print(f"{agent_name}{winner}:")
                print(f"  Score: {player.score}")
                print(f"  Complete Quartets: {quartets if quartets else 'None'}")
                print(f"  Animals: {animals}")
                print(f"  Reward: {env.rewards[agent_name]:.2f}")
            break

    env.close()
    print(f"\nGame completed in {step_count} steps")

if __name__ == "__main__":
    test_visual()