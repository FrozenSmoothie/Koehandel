"""
Test PettingZoo API compliance
"""

from pettingzoo.test import api_test
from koehandel_game_engine import KoehandelPettingZooEnv

def test_api():
    """Test that environment follows PettingZoo API."""
    print("Running PettingZoo API compliance test...")

    env = KoehandelPettingZooEnv(num_players=4)

    try:
        api_test(env, num_cycles=100, verbose_progress=True)
        print("\n[OK] API compliance test passed!")
    except Exception as e:
        print(f"\n[FAIL] API test failed: {e}")
        raise

if __name__ == "__main__":
    test_api()