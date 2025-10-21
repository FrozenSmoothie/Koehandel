"""
Comprehensive test suite for Koehandel environment
"""

import unittest
import numpy as np
from koehandel_game_engine import (
    KoehandelPettingZooEnv,
    KoehandelGame,
    MoneyCard,
    AnimalCard,
    Player
)

class TestMoneyCard(unittest.TestCase):
    """Test MoneyCard class."""

    def test_creation(self):
        card = MoneyCard(50)
        self.assertEqual(card.value, 50)

    def test_repr(self):
        card = MoneyCard(100)
        self.assertIn("100", str(card))

class TestAnimalCard(unittest.TestCase):
    """Test AnimalCard class."""

    def test_creation(self):
        card = AnimalCard("Horse", 1000)
        self.assertEqual(card.animal_type, "Horse")
        self.assertEqual(card.points, 1000)

class TestPlayer(unittest.TestCase):
    """Test Player class."""

    def test_creation(self):
        player = Player("player_0")
        self.assertEqual(player.name, "player_0")
        self.assertEqual(len(player.animal_cards), 0)
        self.assertEqual(len(player.money_cards), 0)

    def test_total_money(self):
        player = Player("player_0")
        player.money_cards = [MoneyCard(10), MoneyCard(50), MoneyCard(100)]
        self.assertEqual(player.total_money(), 160)

    def test_calculate_score(self):
        player = Player("player_0")
        # Give player a complete quartet of horses
        for _ in range(4):
            player.animal_cards.append(AnimalCard("Horse", 1000))

        quartet_values = {"Horse": 1000}
        score = player.calculate_score(quartet_values)
        self.assertEqual(score, 1000)

class TestKoehandelGame(unittest.TestCase):
    """Test KoehandelGame class."""

    def test_initialization(self):
        game = KoehandelGame(["player_0", "player_1"], seed=42)
        self.assertEqual(len(game.players), 2)
        self.assertEqual(len(game.animal_deck), 40)  # 10 types × 4 cards

    def test_starting_money(self):
        game = KoehandelGame(["player_0"], seed=42)
        player = game.players["player_0"]
        self.assertEqual(len(player.money_cards), 7)
        self.assertEqual(player.total_money(), 90)  # 2×0 + 4×10 + 1×50

    def test_auction_start(self):
        game = KoehandelGame(["player_0", "player_1"], seed=42)
        initial_deck_size = len(game.animal_deck)

        card = game.start_auction()

        self.assertIsNotNone(card)
        self.assertEqual(len(game.animal_deck), initial_deck_size - 1)
        self.assertEqual(game.current_auction_card, card)

    def test_possible_bid_amounts(self):
        game = KoehandelGame(["player_0"], seed=42)
        player = game.players["player_0"]

        # Player starts with 2×0 + 4×10 + 1×50 = 90 total
        possible_bids = game._get_possible_bid_amounts(player)

        # Should be able to bid many combinations
        self.assertIn(10, possible_bids)
        self.assertIn(20, possible_bids)
        self.assertIn(50, possible_bids)
        self.assertIn(60, possible_bids)  # 10 + 50
        self.assertIn(90, possible_bids)  # All money

    def test_game_over(self):
        game = KoehandelGame(["player_0"], seed=42)
        self.assertFalse(game.is_game_over())

        # Empty the deck
        game.animal_deck = []
        self.assertTrue(game.is_game_over())

class TestKoehandelEnv(unittest.TestCase):
    """Test KoehandelPettingZooEnv class."""

    def test_initialization(self):
        env = KoehandelPettingZooEnv(num_players=4)
        self.assertEqual(len(env.possible_agents), 4)
        self.assertEqual(env.num_players, 4)

    def test_reset(self):
        env = KoehandelPettingZooEnv(num_players=3)
        env.reset(seed=42)  # Returns None in AECEnv

        self.assertEqual(len(env.agents), 3)
        self.assertTrue(env.has_reset)
        self.assertIsNotNone(env.game)

    def test_observation_shape(self):
        env = KoehandelPettingZooEnv(num_players=4)
        env.reset(seed=42)

        obs = env._get_obs("player_0")
        self.assertEqual(obs.shape, (18,))
        self.assertEqual(obs.dtype, np.float32)

    def test_action_space(self):
        env = KoehandelPettingZooEnv(num_players=4)
        env.reset(seed=42)

        action_space = env.action_space("player_0")  # Call as method
        self.assertEqual(action_space.n, 1000)

    def test_step(self):
        env = KoehandelPettingZooEnv(num_players=4)
        env.reset(seed=42)

        # Take one step
        env.step(0)  # Pass

        # Check that turn advanced
        self.assertGreater(env.game.turn, 0)

    def test_full_game(self):
        """Run a full game to completion."""
        env = KoehandelPettingZooEnv(num_players=4)
        env.reset(seed=42)

        steps = 0
        max_steps = 500

        for agent in env.agent_iter(max_iter=max_steps):
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                action = 0  # Always pass for quick game

            env.step(action)
            steps += 1

            if all(env.terminations.values()):
                break

        # Game should have ended
        self.assertTrue(all(env.terminations.values()))
        print(f"Game completed in {steps} steps")

def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    run_tests()