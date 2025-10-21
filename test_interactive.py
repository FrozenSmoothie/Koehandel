"""
Interactive test - play the game manually
"""

from koehandel_game_engine import KoehandelPettingZooEnv
import numpy as np

def print_player_state(env, agent):
    """Print current player's state."""
    player = env.game.players[agent]
    print(f"\n{'='*60}")
    print(f"{agent}'s turn")
    print(f"{'='*60}")

    # Show animals
    animals = player.get_animal_counts()
    print(f"Animals: {animals}")

    # Show money
    money_cards = sorted([card.value for card in player.money_cards], reverse=True)
    print(f"Money cards: {money_cards}")
    print(f"Total money: ${player.total_money()}")

    # Show current auction
    if env.game.current_auction_card:
        print(f"\nCurrent auction: {env.game.current_auction_card}")
        print(f"Current bids: {env.game.auction_bids}")

    # Show valid actions
    valid_actions = env.game.get_valid_actions(agent)
    print(f"\nValid actions: {len(valid_actions)} options")
    print(f"  0 = Pass")
    if len(valid_actions) > 1:
        print(f"  Available bid amounts: {sorted(valid_actions[1:])[:20]}")

def get_action_input(valid_actions):
    """Get action from user input."""
    while True:
        try:
            action = int(input("\nEnter action (or -1 to see valid actions): "))
            if action == -1:
                print(f"Valid actions: {valid_actions[:50]}")
                continue
            if action in valid_actions or action == 0:
                return action
            else:
                print(f"Invalid action. Choose from valid actions.")
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            print("\nExiting...")
            return 0

def test_interactive():
    """Interactive test mode."""
    print("Starting interactive Koehandel game...")
    print("You'll control all players and make decisions\n")

    env = KoehandelPettingZooEnv(num_players=4, render_mode="human")
    env.reset(seed=42)

    step_count = 0
    max_steps = 200

    try:
        for agent in env.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                print(f"\n{agent} is done. Reward: {reward}")
                if info:
                    print(f"Info: {info}")
                action = None
            else:
                print_player_state(env, agent)

                # Auto-play or manual?
                mode = input("\n[A]uto play random, [M]anual, [Q]uit? ").lower()

                if mode == 'q':
                    break
                elif mode == 'a':
                    valid = env.game.get_valid_actions(agent)
                    action = np.random.choice(valid) if valid else 0
                    print(f"Auto-selected action: {action}")
                else:
                    valid = env.game.get_valid_actions(agent)
                    action = get_action_input(valid)

            env.step(action)
            step_count += 1

            if all(env.terminations.values()):
                print("\n" + "="*60)
                print("GAME OVER!")
                print("="*60)
                for agent_name in env.agents:
                    player = env.game.players[agent_name]
                    print(f"{agent_name}: Score = {player.score}, "
                          f"Reward = {env.rewards[agent_name]:.2f}")
                break
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")

    env.close()
    print(f"\nGame finished in {step_count} steps")

if __name__ == "__main__":
    test_interactive()