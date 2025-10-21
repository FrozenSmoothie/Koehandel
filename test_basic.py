"""
Basic smoke test for Koehandel environment
"""

from koehandel_game_engine import KoehandelPettingZooEnv

def test_basic():
    """Test basic environment functionality."""
    print("Creating environment...")
    env = KoehandelPettingZooEnv(num_players=4, render_mode="human")

    print("Resetting environment...")
    env.reset(seed=42)

    print(f"[OK] Environment created with {len(env.agents)} agents")
    print(f"[OK] Observation shape: {env.observe(env.agents[0]).shape}")
    print(f"[OK] Action space: {env.action_space(env.agents[0])}")

    print("\nRunning 10 random steps...")
    for i in range(10):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                # Take random action
                action = env.action_space(agent).sample()

            env.step(action)

            if all(env.terminations.values()):
                print(f"\n[OK] Game ended after {i} rounds")
                break

        if all(env.terminations.values()):
            break

    env.close()
    print("\n[OK] All basic tests passed!")

if __name__ == "__main__":
    test_basic()