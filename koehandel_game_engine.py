"""
Koehandel - Enhanced Reward Shaping
- Time bonuses for fast quartets
- Diversity rewards for collecting varied animals
- Trade cooldowns to prevent loops
- Diminishing returns on single-quartet focus
"""

import random
from typing import List, Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector


class MoneyCard:
    """Represents a money card with a specific value."""
    def __init__(self, value: int):
        self.value = value

    def __repr__(self):
        return f"Money({self.value})"


class AnimalCard:
    """Represents an animal card with type and point value."""
    def __init__(self, animal_type: str, points: int):
        self.animal_type = animal_type
        self.points = points

    def __repr__(self):
        return f"{self.animal_type}({self.points})"


class Player:
    """Represents a player in the game."""
    def __init__(self, name: str):
        self.name = name
        self.animal_cards: List[AnimalCard] = []
        self.money_cards: List[MoneyCard] = []
        self.score: int = 0
        self.estimated_opponent_money: Dict[str, int] = {}

        # Progress tracking
        self.quartets_completed: int = 0
        self.quartets_completion_turns: Dict[str, int] = {}  # When each quartet completed
        self.cards_won_in_auctions: int = 0
        self.successful_trades: int = 0
        self.money_spent: int = 0

        # Trade cooldown tracking
        self.last_trade_turn: int = -10
        self.trades_this_round: int = 0

    def get_animal_counts(self) -> Dict[str, int]:
        """Returns count of each animal type owned."""
        counts = {}
        for card in self.animal_cards:
            counts[card.animal_type] = counts.get(card.animal_type, 0) + 1
        return counts

    def get_money_counts(self) -> Dict[int, int]:
        """Returns count of each money value owned."""
        counts = {}
        for card in self.money_cards:
            counts[card.value] = counts.get(card.value, 0) + 1
        return counts

    def total_money(self) -> int:
        """Returns total money value."""
        return sum(card.value for card in self.money_cards)

    def calculate_score(self, quartet_values: Dict[str, int]) -> int:
        """Calculate final score based on complete quartets."""
        counts = self.get_animal_counts()
        score = 0
        quartets = 0
        for animal_type, count in counts.items():
            if count == 4:
                score += quartet_values[animal_type]
                quartets += 1
        self.score = score
        self.quartets_completed = quartets
        return score

    def get_diversity_score(self) -> float:
        """
        Calculate diversity bonus - rewards having cards from many different animals.
        Discourages hyper-focusing on just one quartet.
        """
        counts = self.get_animal_counts()

        # Number of different animal types
        num_types = len(counts)

        # Bonus for spreading across types
        diversity = num_types * 0.5

        # Penalty for imbalance (having 4 of one thing and 0 of others)
        if num_types > 0:
            avg_count = sum(counts.values()) / num_types
            variance = sum((count - avg_count) ** 2 for count in counts.values()) / num_types
            balance_bonus = max(0, 2.0 - variance * 0.1)
            diversity += balance_bonus

        return diversity

    def get_progress_score(self, quartet_values: Dict[str, int]) -> float:
        """
        Calculate progress-based score with DIMINISHING RETURNS.
        """
        counts = self.get_animal_counts()
        progress = 0.0

        for animal_type, count in counts.items():
            base_value = quartet_values[animal_type]

            if count == 4:
                # Complete quartet
                progress += base_value
            elif count == 3:
                # Diminishing: 40% instead of 50%
                progress += base_value * 0.4
            elif count == 2:
                # Diminishing: 15% instead of 20%
                progress += base_value * 0.15
            elif count == 1:
                # Small reward for starting
                progress += base_value * 0.05

        return progress


class KoehandelGame:
    """Core game logic for Koehandel with enhanced reward shaping."""

    ANIMAL_TYPES = [
        ("Horse", 1000), ("Cow", 800), ("Pig", 650), ("Donkey", 500),
        ("Goat", 350), ("Sheep", 250), ("Dog", 160), ("Cat", 90),
        ("Goose", 40), ("Rooster", 10)
    ]

    STARTING_MONEY = [0, 0, 10, 10, 10, 10, 50]
    MONEY_VALUES = [0, 10, 50, 100, 200, 500]

    # Game balance parameters
    TRADE_COOLDOWN_TURNS = 3  # Must wait 3 turns between trades
    MAX_TRADES_PER_ROUND = 2  # Max 2 trades per trading round

    def __init__(self, agent_names: List[str], seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.players: Dict[str, Player] = {name: Player(name) for name in agent_names}
        self.agent_names = agent_names
        self.animal_deck: List[AnimalCard] = self._init_animal_deck()
        self.quartet_values: Dict[str, int] = dict(self.ANIMAL_TYPES)
        self.turn: int = 0
        self.donkey_count: int = 0
        self.donkey_payouts: List[int] = [50, 100, 200, 500]

        self.phase: str = "auction"
        self.current_auction_card: Optional[AnimalCard] = None
        self.auction_bids: Dict[str, int] = {}
        self.auction_active_players: List[str] = []

        self.trade_initiator: Optional[str] = None
        self.trade_target: Optional[str] = None
        self.trade_animal_type: Optional[str] = None
        self.trade_mode: Optional[str] = None
        self.trade_bids: Dict[str, int] = {}
        self.trade_bids_submitted: Dict[str, bool] = {}

        self.players_passed_trading: List[str] = []
        self.consecutive_passes_in_trading: int = 0

        # Enhanced tracking
        self.previous_progress: Dict[str, float] = {}
        self.previous_diversity: Dict[str, float] = {}
        self.quartets_completed_log: List[Tuple[str, str, int]] = []  # (player, animal, turn)

        for player in self.players.values():
            player.money_cards = [MoneyCard(v) for v in self.STARTING_MONEY]
            for other_name in agent_names:
                if other_name != player.name:
                    player.estimated_opponent_money[other_name] = 90
            self.previous_progress[player.name] = 0.0
            self.previous_diversity[player.name] = 0.0

    def _init_animal_deck(self) -> List[AnimalCard]:
        """Initialize and shuffle the animal deck."""
        deck = []
        for animal_type, points in self.ANIMAL_TYPES:
            for _ in range(4):
                deck.append(AnimalCard(animal_type, points))
        random.shuffle(deck)
        return deck

    def calculate_step_rewards(self, action_agent: str, action: int) -> Dict[str, float]:
        """
        Calculate ENHANCED dense rewards:
        - Progress rewards (with diminishing returns)
        - Diversity rewards (encourage varied collection)
        - Time bonuses (faster quartets = better)
        - Action rewards (encourage participation)
        - Trade cooldowns (prevent spam)
        """
        rewards = {agent: 0.0 for agent in self.agent_names}

        # Very small time penalty (encourage efficiency but not critical)
        for agent in self.agent_names:
            rewards[agent] -= 0.00005

        # Progress and diversity rewards for ALL players
        for agent_name, player in self.players.items():
            # 1. PROGRESS REWARD (with diminishing returns built-in)
            current_progress = player.get_progress_score(self.quartet_values)
            previous_progress = self.previous_progress[agent_name]
            progress_delta = current_progress - previous_progress

            if progress_delta > 0:
                # Scale: normalize by max single animal (Horse = 1000)
                rewards[agent_name] += progress_delta / 200.0  # Significant reward

            self.previous_progress[agent_name] = current_progress

            # 2. DIVERSITY REWARD (encourage collecting multiple types)
            current_diversity = player.get_diversity_score()
            previous_diversity = self.previous_diversity[agent_name]
            diversity_delta = current_diversity - previous_diversity

            if diversity_delta > 0:
                rewards[agent_name] += diversity_delta * 0.05  # Bonus for diversifying

            self.previous_diversity[agent_name] = current_diversity

            # 3. QUARTET COMPLETION BONUS (with time bonus)
            counts = player.get_animal_counts()
            for animal_type, count in counts.items():
                if count == 4:
                    # Check if this is a NEW quartet
                    if animal_type not in player.quartets_completion_turns:
                        player.quartets_completion_turns[animal_type] = self.turn
                        self.quartets_completed_log.append((agent_name, animal_type, self.turn))

                        base_value = self.quartet_values[animal_type]

                        # BASE REWARD: Normalized quartet value
                        quartet_reward = base_value / 100.0

                        # TIME BONUS: Earlier completion = bigger bonus
                        # Game typically lasts ~200 turns, so:
                        # Turn 50: +50% bonus
                        # Turn 100: +25% bonus
                        # Turn 150: +10% bonus
                        max_game_length = 300
                        turn_ratio = self.turn / max_game_length
                        time_bonus = max(0, 1.0 - turn_ratio)  # 1.0 at start, 0.0 at end

                        quartet_reward *= (1.0 + time_bonus)

                        rewards[agent_name] += quartet_reward

                        # Broadcast info
                        print(f"  🎉 {agent_name} completed {animal_type} quartet at turn {self.turn}! Bonus: {quartet_reward:.2f}")

        # 4. ACTION-SPECIFIC REWARDS
        player = self.players[action_agent]

        if action == 0:  # Pass
            rewards[action_agent] -= 0.01  # Small penalty for passing

        elif 1 <= action <= 999:  # Bidding
            rewards[action_agent] += 0.02  # Reward participation

        elif action >= 1000:  # Initiating trade
            # Check trade cooldown
            if self.turn - player.last_trade_turn < self.TRADE_COOLDOWN_TURNS:
                # Penalty for trading too frequently (spam prevention)
                rewards[action_agent] -= 0.1
            elif player.trades_this_round >= self.MAX_TRADES_PER_ROUND:
                # Penalty for exceeding trade limit
                rewards[action_agent] -= 0.1
            else:
                # Good trade initiation
                rewards[action_agent] += 0.03

        return rewards

    def start_auction(self) -> Optional[AnimalCard]:
        """Draw a card and start an auction."""
        if not self.animal_deck:
            return None

        card = self.animal_deck.pop(0)
        self.current_auction_card = card
        self.auction_active_players = self.agent_names.copy()
        self.auction_bids = {name: 0 for name in self.agent_names}

        if card.animal_type == "Donkey" and self.donkey_count < 4:
            payout = self.donkey_payouts[self.donkey_count]
            for player in self.players.values():
                player.money_cards.append(MoneyCard(payout))
                for other_player in self.players.values():
                    if other_player.name != player.name:
                        other_player.estimated_opponent_money[player.name] += payout
            self.donkey_count += 1

        return card

    def get_possible_trades(self, agent_name: str) -> List[Tuple[str, str, str]]:
        """Get possible trades for an agent (with cooldown enforcement)."""
        player = self.players[agent_name]

        # Enforce trade cooldown
        if self.turn - player.last_trade_turn < self.TRADE_COOLDOWN_TURNS:
            return []

        # Enforce max trades per round
        if player.trades_this_round >= self.MAX_TRADES_PER_ROUND:
            return []

        player_animals = player.get_animal_counts()
        possible_trades = []

        for target_name in self.agent_names:
            if target_name == agent_name:
                continue

            target = self.players[target_name]
            target_animals = target.get_animal_counts()

            for animal_type in player_animals:
                if animal_type in target_animals:
                    player_count = player_animals[animal_type]
                    target_count = target_animals[animal_type]

                    if player_count >= 1 and target_count >= 1:
                        possible_trades.append((target_name, animal_type, "1v1"))

                    if player_count >= 2 and target_count >= 2:
                        possible_trades.append((target_name, animal_type, "2v2"))

        return possible_trades

    def get_valid_actions(self, agent_name: str) -> List[int]:
        """Returns list of valid action indices."""
        valid = [0]

        if self.phase == "auction":
            player = self.players[agent_name]

            if agent_name in self.auction_active_players:
                current_high_bid = max(self.auction_bids.values()) if self.auction_bids else 0
                possible_bids = self._get_possible_bid_amounts(player)

                for bid_amount in possible_bids:
                    if bid_amount > current_high_bid:
                        action_idx = self._bid_amount_to_action(bid_amount)
                        valid.append(action_idx)

        elif self.phase == "trade":
            if self.trade_initiator is None:
                if agent_name not in self.players_passed_trading:
                    possible_trades = self.get_possible_trades(agent_name)
                    for idx, trade in enumerate(possible_trades):
                        valid.append(1000 + idx)

            elif agent_name in [self.trade_initiator, self.trade_target]:
                if not self.trade_bids_submitted.get(agent_name, False):
                    player = self.players[agent_name]
                    possible_bids = self._get_possible_bid_amounts(player)

                    for bid_amount in possible_bids:
                        action_idx = self._bid_amount_to_action(bid_amount)
                        valid.append(action_idx)

        return valid

    def get_action_mask(self, agent_name: str) -> np.ndarray:
        """Returns action mask (1 = valid, 0 = invalid)."""
        mask = np.zeros(2000, dtype=np.int8)
        valid_actions = self.get_valid_actions(agent_name)
        mask[valid_actions] = 1
        return mask

    def _get_possible_bid_amounts(self, player: Player) -> List[int]:
        """Calculate all possible bid amounts."""
        money_values = [card.value for card in player.money_cards]

        if not money_values:
            return []

        possible_sums = {0}

        for value in money_values:
            new_sums = set()
            for existing_sum in possible_sums:
                new_sums.add(existing_sum + value)
            possible_sums.update(new_sums)

        possible_sums.discard(0)
        return sorted(list(possible_sums))

    def _bid_amount_to_action(self, bid_amount: int) -> int:
        return min(bid_amount, 999)

    def _action_to_bid_amount(self, action: int) -> int:
        return action

    def apply_action(self, agent_name: str, action: int) -> Dict[str, float]:
        """Apply an action and return step rewards."""
        if self.phase == "auction":
            self._apply_auction_action(agent_name, action)
        elif self.phase == "trade":
            self._apply_trade_action(agent_name, action)

        return self.calculate_step_rewards(agent_name, action)

    def _apply_auction_action(self, agent_name: str, action: int):
        """Apply action during auction phase."""
        if action == 0:
            if agent_name in self.auction_active_players:
                self.auction_active_players.remove(agent_name)

            if len(self.auction_active_players) <= 1:
                self._resolve_auction()

        elif 1 <= action <= 999:
            bid_amount = self._action_to_bid_amount(action)
            player = self.players[agent_name]

            if self._can_afford_bid(player, bid_amount):
                self.auction_bids[agent_name] = bid_amount
            else:
                if agent_name in self.auction_active_players:
                    self.auction_active_players.remove(agent_name)

    def _apply_trade_action(self, agent_name: str, action: int):
        """Apply action during trade phase."""
        if action == 0:
            if self.trade_initiator is None:
                self.players_passed_trading.append(agent_name)
                self.consecutive_passes_in_trading += 1

                if len(self.players_passed_trading) >= len(self.agent_names) or self.consecutive_passes_in_trading >= 20:
                    self.phase = "game_over"

            elif agent_name in [self.trade_initiator, self.trade_target]:
                self.trade_bids[agent_name] = 0
                self.trade_bids_submitted[agent_name] = True

                if all(self.trade_bids_submitted.get(p, False) for p in [self.trade_initiator, self.trade_target]):
                    self._resolve_trade()

        elif action >= 1000:
            if self.trade_initiator is None:
                trade_idx = action - 1000
                possible_trades = self.get_possible_trades(agent_name)

                if 0 <= trade_idx < len(possible_trades):
                    target, animal_type, mode = possible_trades[trade_idx]
                    self.trade_initiator = agent_name
                    self.trade_target = target
                    self.trade_animal_type = animal_type
                    self.trade_mode = mode
                    self.trade_bids = {}
                    self.trade_bids_submitted = {self.trade_initiator: False, self.trade_target: False}
                    self.consecutive_passes_in_trading = 0

                    # Update trade tracking
                    self.players[agent_name].last_trade_turn = self.turn
                    self.players[agent_name].trades_this_round += 1

        elif 1 <= action <= 999:
            if agent_name in [self.trade_initiator, self.trade_target]:
                if not self.trade_bids_submitted.get(agent_name, False):
                    bid_amount = self._action_to_bid_amount(action)
                    player = self.players[agent_name]

                    if self._can_afford_bid(player, bid_amount):
                        self.trade_bids[agent_name] = bid_amount
                        self.trade_bids_submitted[agent_name] = True

                        if all(self.trade_bids_submitted.get(p, False) for p in [self.trade_initiator, self.trade_target]):
                            self._resolve_trade()

    def _can_afford_bid(self, player: Player, amount: int) -> bool:
        return player.total_money() >= amount

    def _resolve_auction(self):
        """Resolve the current auction."""
        if not self.auction_active_players:
            self.current_auction_card = None
            self.phase = "auction"
            return

        winner = self.auction_active_players[0]
        winning_bid = self.auction_bids[winner]

        self.players[winner].animal_cards.append(self.current_auction_card)
        self.players[winner].cards_won_in_auctions += 1
        self.players[winner].money_spent += winning_bid

        self._pay_money(winner, winning_bid)

        for player in self.players.values():
            if player.name != winner:
                player.estimated_opponent_money[winner] -= winning_bid

        self.current_auction_card = None
        self.phase = "auction"

    def _resolve_trade(self):
        """Resolve the current trade."""
        initiator_bid = self.trade_bids.get(self.trade_initiator, 0)
        target_bid = self.trade_bids.get(self.trade_target, 0)

        if initiator_bid > target_bid:
            winner = self.trade_initiator
            loser = self.trade_target
        elif target_bid > initiator_bid:
            winner = self.trade_target
            loser = self.trade_initiator
        else:
            winner = self.trade_initiator
            loser = self.trade_target

        winner_player = self.players[winner]
        loser_player = self.players[loser]

        winner_player.successful_trades += 1

        num_cards = 1 if self.trade_mode == "1v1" else 2
        cards_to_transfer = []

        for card in loser_player.animal_cards[:]:
            if card.animal_type == self.trade_animal_type and len(cards_to_transfer) < num_cards:
                cards_to_transfer.append(card)
                loser_player.animal_cards.remove(card)

        for card in cards_to_transfer:
            winner_player.animal_cards.append(card)

        self._pay_money(winner, self.trade_bids.get(winner, 0))

        # Reset trade state
        self.trade_initiator = None
        self.trade_target = None
        self.trade_animal_type = None
        self.trade_mode = None
        self.trade_bids = {}
        self.trade_bids_submitted = {}
        self.players_passed_trading = []
        self.consecutive_passes_in_trading = 0

    def start_new_trading_round(self):
        """Start a new trading round (reset per-round counters)."""
        for player in self.players.values():
            player.trades_this_round = 0

    def _pay_money(self, agent_name: str, amount: int):
        """Remove money cards from player's hand."""
        player = self.players[agent_name]
        remaining = amount
        money_to_remove = []

        sorted_money = sorted(player.money_cards, key=lambda x: x.value, reverse=True)

        for card in sorted_money:
            if remaining > 0 and card.value <= remaining:
                money_to_remove.append(card)
                remaining -= card.value

        if remaining > 0 and amount <= player.total_money():
            money_to_remove = self._find_optimal_payment(player.money_cards, amount)

        for card in money_to_remove:
            if card in player.money_cards:
                player.money_cards.remove(card)

    def _find_optimal_payment(self, money_cards: List[MoneyCard], target: int) -> List[MoneyCard]:
        """Find optimal combination of money cards."""
        n = len(money_cards)

        if n == 0:
            return []

        dp = [[False] * (target + 1) for _ in range(n + 1)]
        parent = [[None] * (target + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = True

        for i in range(1, n + 1):
            card_value = money_cards[i - 1].value
            for j in range(target + 1):
                if dp[i - 1][j]:
                    dp[i][j] = True
                    parent[i][j] = (i - 1, j, False)

                if j >= card_value and dp[i - 1][j - card_value]:
                    dp[i][j] = True
                    parent[i][j] = (i - 1, j - card_value, True)

        best_amount = target
        while best_amount >= 0 and not dp[n][best_amount]:
            best_amount -= 1

        if best_amount < 0:
            return []

        cards_to_use = []
        i, j = n, best_amount

        while i > 0 and j > 0:
            if parent[i][j] is None:
                break
            prev_i, prev_j, took_card = parent[i][j]
            if took_card:
                cards_to_use.append(money_cards[i - 1])
            i, j = prev_i, prev_j

        return cards_to_use

    def is_game_over(self) -> bool:
        """Check if game has ended."""
        if len(self.animal_deck) > 0:
            return False
        return self.phase == "game_over"

    def get_winners(self) -> List[str]:
        """Calculate scores and return winners."""
        scores = {}
        for name, player in self.players.items():
            scores[name] = player.calculate_score(self.quartet_values)

        max_score = max(scores.values())
        return [name for name, score in scores.items() if score == max_score]


class KoehandelPettingZooEnv(AECEnv):
    """PettingZoo AEC Environment for Koehandel with enhanced rewards."""

    metadata = {
        "render_modes": ["human"],
        "name": "koehandel_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None):
        super().__init__()

        self.num_players = num_players
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        # Observation space
        animal_low = [0.0] * 10
        animal_high = [4.0] * 10

        money_low = [0.0] * 6
        money_high = [100.0] * 6

        opponent_money_low = [0.0] * (num_players - 1)
        opponent_money_high = [5000.0] * (num_players - 1)

        game_state_low = [0.0, 0.0, 0.0]
        game_state_high = [40.0, 10000.0, 2.0]

        obs_size = 10 + 6 + (num_players - 1) + 3

        low_bounds = np.array(animal_low + money_low + opponent_money_low + game_state_low, dtype=np.float32)
        high_bounds = np.array(animal_high + money_high + opponent_money_high + game_state_high, dtype=np.float32)

        single_obs_space = spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32
        )

        self.observation_spaces = {
            agent: single_obs_space for agent in self.possible_agents
        }

        single_action_space = spaces.Discrete(2000)
        self.action_spaces = {
            agent: single_action_space for agent in self.possible_agents
        }

        self.game: Optional[KoehandelGame] = None
        self.agents = []
        self._agent_selector = None
        self.agent_selection = None

        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

        self.has_reset = False

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observe(self, agent: str) -> np.ndarray:
        return self._get_obs(agent)

    def _get_obs(self, agent_name: str) -> np.ndarray:
        """Generate observation vector for an agent."""
        if self.game is None:
            obs_size = 10 + 6 + (self.num_players - 1) + 3
            return np.zeros(obs_size, dtype=np.float32)

        player = self.game.players[agent_name]

        animal_counts = []
        for animal_type, _ in KoehandelGame.ANIMAL_TYPES:
            count = sum(1 for card in player.animal_cards if card.animal_type == animal_type)
            animal_counts.append(float(count))

        money_counts = []
        for value in KoehandelGame.MONEY_VALUES:
            count = sum(1 for card in player.money_cards if card.value == value)
            money_counts.append(float(count))

        opponent_money_estimates = []
        for other_name in self.game.agent_names:
            if other_name != agent_name:
                estimated = player.estimated_opponent_money.get(other_name, 90.0)
                opponent_money_estimates.append(float(estimated))

        deck_size = float(len(self.game.animal_deck))
        turn_num = float(self.game.turn)
        phase_num = 0.0
        if self.game.phase == "trade":
            phase_num = 1.0
        elif self.game.phase == "game_over":
            phase_num = 2.0

        obs = np.array(
            animal_counts + money_counts + opponent_money_estimates + [deck_size, turn_num, phase_num],
            dtype=np.float32
        )
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = self.possible_agents[:]
        self.game = KoehandelGame(self.agents, seed=seed)

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.game.start_auction()

        self.has_reset = True

    def step(self, action: int):
        """Execute one step with enhanced dense rewards."""
        if not self.has_reset:
            raise RuntimeError("Environment must be reset before stepping")

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection

        # Validate action
        action_mask = self.game.get_action_mask(agent)
        if action_mask[action] == 0:
            action = 0

        # Get step rewards
        step_rewards = self.game.apply_action(agent, action)
        self.rewards = step_rewards

        self.infos[agent] = {"action_mask": action_mask}

        self.game.turn += 1

        # Check if game is over
        if self.game.is_game_over():
            self._handle_game_end()
        else:
            if len(self.game.animal_deck) == 0 and self.game.phase == "auction":
                self.game.phase = "trade"
                self.game.players_passed_trading = []
                self.game.start_new_trading_round()  # Reset trade counters

            if self.game.phase == "auction" and self.game.current_auction_card is None:
                card = self.game.start_auction()
                if card is None and self.game.phase == "auction":
                    self.game.phase = "trade"
                    self.game.players_passed_trading = []
                    self.game.start_new_trading_round()

        # Accumulate rewards
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

        # Select next agent
        if not all(self.terminations.values()):
            self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def _handle_game_end(self):
        """Handle end of game with final bonuses."""
        winners = self.game.get_winners()

        max_possible_score = sum(points for _, points in KoehandelGame.ANIMAL_TYPES)

        for agent in self.agents:
            player = self.game.players[agent]

            # Base: normalized final score
            normalized_score = player.score / max_possible_score

            # Win bonus
            win_bonus = 3.0 if agent in winners else 0.0

            # Diversity bonus
            diversity_score = player.get_diversity_score()

            # Final reward combines all factors
            self.rewards[agent] = normalized_score * 15.0 + win_bonus + diversity_score

            self.terminations[agent] = True
            self.infos[agent] = {
                "score": player.score,
                "winner": agent in winners,
                "quartets": player.quartets_completed,
                "cards_won": player.cards_won_in_auctions,
                "trades": player.successful_trades,
                "normalized_score": normalized_score,
                "diversity_score": diversity_score,
                "action_mask": np.ones(2000, dtype=np.int8)
            }

    def _was_dead_step(self, action):
        """Handle step when agent is already terminated."""
        if not all(self.terminations.values()):
            self.agent_selection = self._agent_selector.next()

    def render(self):
        """Render the current game state."""
        if self.render_mode != "human":
            return

        print(f"\n{'='*60}")
        print(f"Turn {self.game.turn} | Phase: {self.game.phase.upper()}")
        print(f"Deck remaining: {len(self.game.animal_deck)} cards")

        if self.game.phase == "auction" and self.game.current_auction_card:
            print(f"Current auction: {self.game.current_auction_card}")

        if self.game.phase == "trade":
            if self.game.trade_initiator:
                print(f"Trade: {self.game.trade_initiator} vs {self.game.trade_target}")
            else:
                print(f"Waiting for trade initiation (passes: {self.game.consecutive_passes_in_trading}/20)")

        print(f"\nPlayer Status:")
        for name, player in self.game.players.items():
            animals = player.get_animal_counts()
            quartets = [a for a, c in animals.items() if c == 4]
            diversity = player.get_diversity_score()
            progress = player.get_progress_score(self.game.quartet_values)
            print(f"  {name}: Quartets={quartets}, Progress={progress:.0f}, Diversity={diversity:.1f}")

        print(f"{'='*60}\n")

    def close(self):
        pass