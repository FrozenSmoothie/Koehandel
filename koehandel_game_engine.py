"""
Koehandel - Rules-aligned Game Engine with no-money penalty and guarded bailout

This file implements:
- Auction phase with auctioneer-buy step (auctioneer may match winning bid).
- Trading phase with secret-bid trades and 1v1 / 2v2 rules (2v2 forced if both have exactly 2).
- Forced trading rounds after deck exhaustion until all quartets completed (or safety cap).
- NO_MONEY_PENALTY: per-turn negative reward for being broke to discourage intentional bankruptcy.
- Guarded one-time bailout: if multiple players are broke for several forced rounds, grant
  a single small bailout to each broke player, but apply a one-time immediate penalty
  and a final-game penalty so the AI does not learn to exploit the bailout.
- Action masking and trade_options exposed in infos for policy decoding.
"""
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector

# Tunables
MINIMAL_BID = 10            # extra-bid-round minimum
SAFETY_STIPEND = 10         # value of bailout money card if granted (one-time)
MAX_FORCED_TRADE_ROUNDS = 200
BUY_ACTION = 1999           # auctioneer buy action index
NO_MONEY_PENALTY = 0.05     # per-turn penalty for having zero money
BAILOUT_WAIT_ROUNDS = 3     # wait this many forced rounds before one-time bailout
BAILOUT_IMMEDIATE_PENALTY = 1.0  # one-time negative reward applied when bailout is granted
BAILOUT_FINAL_PENALTY = 2.0      # penalty applied to final reward if bailout was used
QUARTET_TIME_MULT = 1.5     # multiplier for time-weighted quartet reward

# ----------------- Basic data structures ----------------- #
class MoneyCard:
    def __init__(self, value: int):
        self.value = int(value)

    def __repr__(self):
        return f"Money({self.value})"


class AnimalCard:
    def __init__(self, animal_type: str, points: int):
        self.animal_type = animal_type
        self.points = int(points)

    def __repr__(self):
        return f"{self.animal_type}({self.points})"


class Player:
    def __init__(self, name: str):
        self.name = name
        self.animal_cards: List[AnimalCard] = []
        self.money_cards: List[MoneyCard] = []
        self.score: int = 0
        self.estimated_opponent_money: Dict[str, int] = {}

        # tracking
        self.quartets_completed: int = 0
        self.quartets_completion_turns: Dict[str, int] = {}
        self.cards_won_in_auctions: int = 0
        self.successful_trades: int = 0
        self.money_spent: int = 0

        # trade cooldown / per-round
        self.last_trade_turn: int = -10
        self.trades_this_round: int = 0

    def get_animal_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.animal_cards:
            counts[c.animal_type] = counts.get(c.animal_type, 0) + 1
        return counts

    def get_money_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for c in self.money_cards:
            counts[c.value] = counts.get(c.value, 0) + 1
        return counts

    def total_money(self) -> int:
        return sum(c.value for c in self.money_cards)

    def calculate_score(self, quartet_values: Dict[str, int]) -> int:
        """
        Calculate final score for the player according to rule:
          final_score = total_points * number_of_quartets
        where total_points = sum(points_per_quartet * quartets_of_that_type)
        and number_of_quartets = total number of completed quartets the player has.

        This returns an integer final score (0 if no quartets).
        """
        counts = self.get_animal_counts()
        total_points = 0
        quartets = 0
        for animal_type, cnt in counts.items():
            if cnt >= 4:
                pts = quartet_values.get(animal_type, 0)
                total_points += pts
                quartets += 1
        final_score = int(total_points * quartets) if quartets > 0 else 0
        self.score = final_score
        self.quartets_completed = quartets
        return final_score

    def get_diversity_score(self) -> float:
        counts = self.get_animal_counts()
        num_types = len(counts)
        diversity = num_types * 0.5
        if num_types > 0:
            avg = sum(counts.values()) / num_types
            variance = sum((c - avg) ** 2 for c in counts.values()) / num_types
            balance_bonus = max(0.0, 2.0 - variance * 0.1)
            diversity += balance_bonus
        return float(diversity)

    def get_progress_score(self, quartet_values: Dict[str, int]) -> float:
        counts = self.get_animal_counts()
        prog = 0.0
        for animal_type, cnt in counts.items():
            base = quartet_values[animal_type]
            if cnt >= 4:
                prog += base
            elif cnt == 3:
                prog += base * 0.4
            elif cnt == 2:
                prog += base * 0.15
            elif cnt == 1:
                prog += base * 0.05
        return float(prog)


# ----------------- Game core ----------------- #
class KoehandelGame:
    ANIMAL_TYPES = [
        ("Horse", 1000), ("Cow", 800), ("Pig", 650), ("Donkey", 500),
        ("Goat", 350), ("Sheep", 250), ("Dog", 160), ("Cat", 90),
        ("Goose", 40), ("Rooster", 10)
    ]
    STARTING_MONEY = [0, 0, 10, 10, 10, 10, 50]
    MONEY_VALUES = [0, 10, 50, 100, 200, 500]

    TRADE_COOLDOWN_TURNS = 3
    MAX_TRADES_PER_ROUND = 2

    def __init__(self, agent_names: List[str], seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent_names = list(agent_names)
        self.players: Dict[str, Player] = {n: Player(n) for n in self.agent_names}
        self.animal_deck: List[AnimalCard] = self._init_animal_deck()
        self.quartet_values: Dict[str, int] = dict(self.ANIMAL_TYPES)
        self.turn: int = 0

        # auction bookkeeping
        self.current_auction_card: Optional[AnimalCard] = None
        self.auction_bids: Dict[str, int] = {}
        self.auction_active_players: List[str] = []
        self.current_auctioneer: Optional[str] = None
        self.next_auctioneer_idx: int = 0

        # pending auctioneer choice
        self.pending_auction_winner: Optional[str] = None
        self.pending_auction_bid: int = 0

        # trade bookkeeping
        self.phase: str = "auction"  # auction / auctioneer_buy / trade / game_over
        self.trade_initiator: Optional[str] = None
        self.trade_target: Optional[str] = None
        self.trade_animal_type: Optional[str] = None
        self.trade_mode: Optional[str] = None
        self.trade_bids: Dict[str, int] = {}
        self.trade_bids_submitted: Dict[str, bool] = {}
        self.players_passed_trading: List[str] = []
        self.consecutive_passes_in_trading: int = 0

        # forced trading rounds
        self.forced_trading_mode: bool = False
        self.players_passed_this_round: List[str] = []
        self.forced_trade_rounds: int = 0
        self.max_forced_trade_rounds: int = MAX_FORCED_TRADE_ROUNDS
        self.trades_executed_in_current_round: int = 0

        # reward tracking
        self.previous_progress: Dict[str, float] = {}
        self.previous_diversity: Dict[str, float] = {}
        self.quartets_completed_log: List[Tuple[str, str, int]] = []
        self.donkey_count = 0
        self.donkey_payouts = [50, 100, 200, 500]

        # bailout bookkeeping (one-time bailout only after wait rounds)
        self.bailout_granted: Dict[str, bool] = {n: False for n in self.agent_names}
        # flag that a one-time immediate bailout penalty must be applied in next reward step
        self.bailout_immediate_penalty_pending: Dict[str, bool] = {n: False for n in self.agent_names}
        # track if final bailout penalty should be applied
        self.bailout_used: Dict[str, bool] = {n: False for n in self.agent_names}

        # initialize money and estimates
        for p in self.players.values():
            p.money_cards = [MoneyCard(v) for v in self.STARTING_MONEY]
            for other in self.agent_names:
                if other != p.name:
                    p.estimated_opponent_money[other] = 90
            self.previous_progress[p.name] = 0.0
            self.previous_diversity[p.name] = 0.0

    def _init_animal_deck(self) -> List[AnimalCard]:
        deck: List[AnimalCard] = []
        for animal_type, points in self.ANIMAL_TYPES:
            for _ in range(4):
                deck.append(AnimalCard(animal_type, points))
        random.shuffle(deck)
        return deck

    # ---------- Utility / bailout helpers ----------
    def maybe_grant_bailout(self):
        """
        Grant a one-time bailout (SAFETY_STIPEND) to any zero-money players only
        if there are >=2 zero-money players AND forced_trade_rounds >= BAILOUT_WAIT_ROUNDS.
        When granted, mark bailout_granted and bailout_used and set immediate penalty pending
        so reward function applies a one-time negative incentive for using bailout.
        """
        zero_players = [name for name, p in self.players.items() if p.total_money() == 0 and not self.bailout_granted[name]]
        if len([p for p in self.players.values() if p.total_money() == 0]) >= 2 and self.forced_trade_rounds >= BAILOUT_WAIT_ROUNDS:
            # grant only once per player
            for name in zero_players:
                # grant one-money-card of SAFETY_STIPEND
                self.players[name].money_cards.append(MoneyCard(SAFETY_STIPEND))
                self.bailout_granted[name] = True
                self.bailout_used[name] = True
                self.bailout_immediate_penalty_pending[name] = True

    def all_quartets_completed(self) -> bool:
        for animal_type, _ in self.ANIMAL_TYPES:
            found = False
            for p in self.players.values():
                if p.get_animal_counts().get(animal_type, 0) == 4:
                    found = True
                    break
            if not found:
                return False
        return True

    # ---------- Reward shaping ----------
    def calculate_step_rewards(self, action_agent: str, action: int) -> Dict[str, float]:
        rewards: Dict[str, float] = {a: 0.0 for a in self.agent_names}
        # tiny time penalty
        for a in self.agent_names:
            rewards[a] -= 0.00005

        # per-turn zero-money penalty (discourage going broke intentionally)
        for a in self.agent_names:
            if self.players[a].total_money() == 0:
                rewards[a] -= NO_MONEY_PENALTY

        # apply one-time immediate bailout penalty if pending
        for a in self.agent_names:
            if self.bailout_immediate_penalty_pending.get(a, False):
                rewards[a] -= BAILOUT_IMMEDIATE_PENALTY
                # apply once
                self.bailout_immediate_penalty_pending[a] = False

        # progress and diversity deltas + quartet completion bonuses
        for name, player in self.players.items():
            current_progress = player.get_progress_score(self.quartet_values)
            prev = self.previous_progress.get(name, 0.0)
            delta = current_progress - prev
            if delta > 0:
                rewards[name] += float(delta) / 150.0  # stronger progress signal
            self.previous_progress[name] = current_progress

            current_div = player.get_diversity_score()
            prev_div = self.previous_diversity.get(name, 0.0)
            ddelta = current_div - prev_div
            if ddelta > 0:
                rewards[name] += float(ddelta) * 0.08  # stronger diversity incentive
            self.previous_diversity[name] = current_div

            for animal_type, cnt in player.get_animal_counts().items():
                if cnt >= 4 and animal_type not in player.quartets_completion_turns:
                    player.quartets_completion_turns[animal_type] = self.turn
                    self.quartets_completed_log.append((name, animal_type, self.turn))
                    base = self.quartet_values[animal_type]
                    quartet_reward = base / 100.0
                    max_game_length = 300.0
                    turn_ratio = float(self.turn) / max_game_length
                    time_bonus = max(0.0, 1.0 - turn_ratio)
                    quartet_reward *= (1.0 + time_bonus * QUARTET_TIME_MULT)
                    rewards[name] += float(quartet_reward)
                    print(f"  🎉 {name} completed {animal_type} quartet at turn {self.turn}! Bonus: {quartet_reward:.2f}")

        # action-specific incentives/penalties
        player = self.players[action_agent]
        if action == 0:
            rewards[action_agent] -= 0.02
        elif 1 <= action <= 999:
            rewards[action_agent] += 0.01
        elif action >= 1000:
            if self.turn - player.last_trade_turn < self.TRADE_COOLDOWN_TURNS:
                rewards[action_agent] -= 0.1
            elif player.trades_this_round >= self.MAX_TRADES_PER_ROUND:
                rewards[action_agent] -= 0.1
            else:
                rewards[action_agent] += 0.06

        return rewards

    # ---------- Auction methods (with auctioneer buy) ----------
    def start_auction(self) -> Optional[AnimalCard]:
        if not self.animal_deck:
            return None
        card = self.animal_deck.pop(0)
        self.current_auction_card = card
        if len(self.agent_names) == 0:
            return None
        self.current_auctioneer = self.agent_names[self.next_auctioneer_idx % len(self.agent_names)]
        self.next_auctioneer_idx = (self.next_auctioneer_idx + 1) % len(self.agent_names)
        self.auction_active_players = [n for n in self.agent_names if n != self.current_auctioneer]
        self.auction_bids = {name: 0 for name in self.agent_names}
        self.players_passed_trading = []
        self.consecutive_passes_in_trading = 0

        # donkey payout handling
        if card.animal_type == "Donkey" and self.donkey_count < 4:
            payout = self.donkey_payouts[self.donkey_count]
            for p in self.players.values():
                p.money_cards.append(MoneyCard(payout))
                for other in self.players.values():
                    if other.name != p.name:
                        other.estimated_opponent_money[p.name] += payout
            self.donkey_count += 1

        return card

    def _get_highest_bidder(self) -> Optional[str]:
        bids = {n: b for n, b in self.auction_bids.items() if n != self.current_auctioneer}
        if not bids:
            return None
        max_bid = max(bids.values())
        if max_bid <= 0:
            return None
        for n in self.agent_names:
            if n != self.current_auctioneer and self.auction_bids.get(n, 0) == max_bid:
                return n
        return None

    def _resolve_auction(self):
        """
        Determine auction winner among non-auctioneers.
        If we have a winner, set pending_auction_* and enter auctioneer_buy phase.
        If no bids (even after extra round), award to auctioneer immediately.
        """
        if self.current_auction_card is None:
            return

        winner = self._get_highest_bidder()
        if winner is None:
            # extra minimal bid round
            any_extra = False
            for name in [n for n in self.agent_names if n != self.current_auctioneer]:
                p = self.players[name]
                if p.total_money() >= MINIMAL_BID:
                    self.auction_bids[name] = max(self.auction_bids.get(name, 0), MINIMAL_BID)
                    any_extra = True
            if any_extra:
                winner = self._get_highest_bidder()
            else:
                # award to auctioneer
                auct = self.current_auctioneer
                if auct is not None:
                    self.players[auct].animal_cards.append(self.current_auction_card)
                    self.players[auct].cards_won_in_auctions += 1
                self.current_auction_card = None
                self.phase = "auction"
                return

        # We have a non-auctioneer winner
        winning_bid = self.auction_bids.get(winner, 0)
        self.pending_auction_winner = winner
        self.pending_auction_bid = int(winning_bid)
        # move to auctioneer buy decision
        self.phase = "auctioneer_buy"
        # clear active bidders for now
        self.auction_active_players = []

    def finalize_auction_after_auctioneer(self, auctioneer_bought: bool):
        """
        Finalize auction after auctioneer's decision:
        - If auctioneer_bought: auctioneer pays winner the bid and keeps the card.
        - Else: winner pays auctioneer (legacy) and gets card.
        """
        winner = self.pending_auction_winner
        bid = int(self.pending_auction_bid or 0)
        auct = self.current_auctioneer

        if self.current_auction_card is None:
            self.pending_auction_winner = None
            self.pending_auction_bid = 0
            self.phase = "auction"
            return

        if auctioneer_bought and auct is not None:
            # auctioneer pays winner and keeps card
            if bid > 0:
                self._pay_money(auct, bid)
                # give equivalent denominations to winner (approximate)
                remaining = bid
                for denom in sorted(self.MONEY_VALUES, reverse=True):
                    while remaining >= denom and denom > 0:
                        self.players[winner].money_cards.append(MoneyCard(denom))
                        remaining -= denom
            self.players[auct].animal_cards.append(self.current_auction_card)
            self.players[auct].cards_won_in_auctions += 1
        else:
            # winner pays auctioneer and winner gets card
            if bid > 0:
                self.players[winner].money_spent += bid
                self._pay_money(winner, bid)
                if auct is not None:
                    remaining = bid
                    for denom in sorted(self.MONEY_VALUES, reverse=True):
                        while remaining >= denom and denom > 0:
                            self.players[auct].money_cards.append(MoneyCard(denom))
                            remaining -= denom
            self.players[winner].animal_cards.append(self.current_auction_card)
            self.players[winner].cards_won_in_auctions += 1

        # clear pending
        self.current_auction_card = None
        self.pending_auction_winner = None
        self.pending_auction_bid = 0
        self.phase = "auction"
        # enforce money safety if needed (bailout not automatic here)
        self.enforce_minimum_money_if_needed()

    # ---------- Trade / Deal methods ----------
    def get_possible_trades(self, agent_name: str) -> List[Tuple[str, str, str]]:
        player = self.players[agent_name]
        # cooldown and per-round limit
        if self.turn - player.last_trade_turn < self.TRADE_COOLDOWN_TURNS:
            return []
        if player.trades_this_round >= self.MAX_TRADES_PER_ROUND:
            return []
        poss: List[Tuple[str, str, str]] = []
        p_animals = player.get_animal_counts()
        for target_name in self.agent_names:
            if target_name == agent_name:
                continue
            target = self.players[target_name]
            t_animals = target.get_animal_counts()
            for animal_type in p_animals:
                if animal_type in t_animals:
                    pc = p_animals[animal_type]
                    tc = t_animals[animal_type]
                    # if both have exactly 2 -> enforce 2v2
                    if pc == 2 and tc == 2:
                        poss.append((target_name, animal_type, "2v2"))
                    else:
                        if pc >= 1 and tc >= 1:
                            poss.append((target_name, animal_type, "1v1"))
                        if pc >= 2 and tc >= 2:
                            poss.append((target_name, animal_type, "2v2"))
        return poss

    def get_valid_actions(self, agent_name: str) -> List[int]:
        valid = [0]
        if self.phase == "auction":
            if agent_name in self.auction_active_players:
                current_high = max(self.auction_bids.values()) if self.auction_bids else 0
                possible_bids = self._get_possible_bid_amounts(self.players[agent_name])
                for b in possible_bids:
                    if b > current_high:
                        valid.append(self._bid_amount_to_action(b))
        elif self.phase == "auctioneer_buy":
            if agent_name == self.current_auctioneer:
                # allow BUY_ACTION only if auctioneer can afford pending bid
                if self.pending_auction_bid is not None and self.players[agent_name].total_money() >= int(self.pending_auction_bid):
                    valid.append(BUY_ACTION)
        elif self.phase == "trade":
            if self.trade_initiator is None:
                if agent_name not in self.players_passed_trading:
                    possible = self.get_possible_trades(agent_name)
                    for idx in range(len(possible)):
                        valid.append(1000 + idx)
            elif agent_name in [self.trade_initiator, self.trade_target]:
                if not self.trade_bids_submitted.get(agent_name, False):
                    p = self.players[agent_name]
                    poss = self._get_possible_bid_amounts(p)
                    for b in poss:
                        valid.append(self._bid_amount_to_action(b))
        return valid

    def get_action_mask(self, agent_name: str) -> np.ndarray:
        mask = np.zeros(2000, dtype=np.int8)
        for idx in self.get_valid_actions(agent_name):
            if 0 <= idx < mask.size:
                mask[idx] = 1
        return mask

    def _get_possible_bid_amounts(self, player: Player) -> List[int]:
        money_values = [c.value for c in player.money_cards]
        if not money_values:
            return []
        possible = {0}
        for v in money_values:
            new = set()
            for s in possible:
                new.add(s + v)
            possible |= new
        possible.discard(0)
        return sorted(possible)

    def _bid_amount_to_action(self, bid_amount: int) -> int:
        return min(int(bid_amount), 999)

    def _action_to_bid_amount(self, action: int) -> int:
        return int(action)

    def apply_action(self, agent_name: str, action: int) -> Dict[str, float]:
        # Route by phase
        if self.phase == "auction":
            self._apply_auction_action(agent_name, action)
        elif self.phase == "auctioneer_buy":
            # only auctioneer should be making this decision
            if agent_name == self.current_auctioneer:
                if action == BUY_ACTION:
                    self.finalize_auction_after_auctioneer(auctioneer_bought=True)
                else:
                    self.finalize_auction_after_auctioneer(auctioneer_bought=False)
                # after finalizing, consider maybe granting bailouts if forced rounds progressed
                self.maybe_grant_bailout()
        elif self.phase == "trade":
            self._apply_trade_action(agent_name, action)
        return self.calculate_step_rewards(agent_name, int(action))

    def _apply_auction_action(self, agent_name: str, action: int):
        # enforce no-money must pass
        if action != 0 and 1 <= action <= 999:
            player = self.players[agent_name]
            bid_amount = self._action_to_bid_amount(action)
            if player.total_money() < bid_amount:
                action = 0
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
        # passing when no active trade
        if action == 0:
            if self.trade_initiator is None:
                if agent_name not in self.players_passed_trading:
                    self.players_passed_trading.append(agent_name)
                if self.forced_trading_mode:
                    if agent_name not in self.players_passed_this_round:
                        self.players_passed_this_round.append(agent_name)
                    if len(self.players_passed_this_round) >= len(self.agent_names):
                        self.forced_trade_rounds += 1
                        # after a full forced round, allow bailout check
                        self.maybe_grant_bailout()
                        if self.all_quartets_completed():
                            self.phase = "game_over"
                        elif self.forced_trade_rounds >= self.max_forced_trade_rounds:
                            self.phase = "game_over"
                        else:
                            self.players_passed_this_round = []
                            self.players_passed_trading = []
                            self.trades_executed_in_current_round = 0
                            self.start_new_trading_round()
                else:
                    if len(self.players_passed_trading) >= len(self.agent_names) or self.consecutive_passes_in_trading >= 20:
                        self.phase = "game_over"
            elif agent_name in [self.trade_initiator, self.trade_target]:
                self.trade_bids[agent_name] = 0
                self.trade_bids_submitted[agent_name] = True
                if all(self.trade_bids_submitted.get(p, False) for p in [self.trade_initiator, self.trade_target]):
                    self._resolve_trade()

        # initiate trade
        elif action >= 1000:
            if self.trade_initiator is None:
                trade_idx = int(action) - 1000
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
                    self.players[agent_name].last_trade_turn = self.turn
                    self.players[agent_name].trades_this_round += 1
                    if self.forced_trading_mode:
                        self.trades_executed_in_current_round += 1
                        self.players_passed_this_round = []

        # submit secret bid
        elif 1 <= action <= 999:
            if agent_name in [self.trade_initiator, self.trade_target]:
                if not self.trade_bids_submitted.get(agent_name, False):
                    bid_amount = self._action_to_bid_amount(action)
                    player = self.players[agent_name]
                    if player.total_money() < bid_amount:
                        # must pass
                        self.trade_bids[agent_name] = 0
                        self.trade_bids_submitted[agent_name] = True
                    else:
                        self.trade_bids[agent_name] = bid_amount
                        self.trade_bids_submitted[agent_name] = True
                    if all(self.trade_bids_submitted.get(p, False) for p in [self.trade_initiator, self.trade_target]):
                        self._resolve_trade()

    def _can_afford_bid(self, player: Player, amount: int) -> bool:
        return player.total_money() >= amount

    def _resolve_trade(self):
        init = self.trade_initiator
        targ = self.trade_target
        if init is None or targ is None:
            self.trade_initiator = None
            self.trade_target = None
            self.trade_animal_type = None
            self.trade_mode = None
            self.trade_bids = {}
            self.trade_bids_submitted = {}
            return

        a_bid = int(self.trade_bids.get(init, 0))
        b_bid = int(self.trade_bids.get(targ, 0))

        # If both have zero money -> attacker wins
        if self.players[init].total_money() == 0 and self.players[targ].total_money() == 0:
            winner, loser = init, targ
        else:
            if a_bid > b_bid:
                winner, loser = init, targ
            elif b_bid > a_bid:
                winner, loser = targ, init
            else:
                # tie -> attacker wins
                winner, loser = init, targ

        winner_player = self.players[winner]
        loser_player = self.players[loser]

        winner_player.successful_trades += 1

        num_cards = 1 if self.trade_mode == "1v1" else 2
        cards_to_transfer: List[AnimalCard] = []
        for card in loser_player.animal_cards[:]:
            if card.animal_type == self.trade_animal_type and len(cards_to_transfer) < num_cards:
                cards_to_transfer.append(card)
                loser_player.animal_cards.remove(card)
        for card in cards_to_transfer:
            winner_player.animal_cards.append(card)

        # exchange bids (approximate)
        def deduct_money(player_name: str, amount: int):
            if amount <= 0:
                return
            pl = self.players[player_name]
            remaining = amount
            pl.money_cards.sort(key=lambda c: c.value, reverse=True)
            to_remove: List[MoneyCard] = []
            for c in pl.money_cards[:]:
                if remaining <= 0:
                    break
                if c.value <= remaining and c.value > 0:
                    to_remove.append(c)
                    remaining -= c.value
            if remaining > 0 and sum(c.value for c in pl.money_cards) >= amount:
                for c in pl.money_cards[:]:
                    if c not in to_remove and remaining > 0:
                        to_remove.append(c)
                        remaining -= c.value
            for c in to_remove:
                if c in pl.money_cards:
                    pl.money_cards.remove(c)

        deduct_money(init, a_bid)
        deduct_money(targ, b_bid)

        def give_money(player_name: str, amount: int):
            remaining = amount
            for denom in sorted(self.MONEY_VALUES, reverse=True):
                while remaining >= denom and denom > 0:
                    self.players[player_name].money_cards.append(MoneyCard(denom))
                    remaining -= denom

        give_money(init, b_bid)
        give_money(targ, a_bid)

        # reset trade state
        self.trade_initiator = None
        self.trade_target = None
        self.trade_animal_type = None
        self.trade_mode = None
        self.trade_bids = {}
        self.trade_bids_submitted = {}
        self.players_passed_trading = []
        self.consecutive_passes_in_trading = 0

        if self.forced_trading_mode:
            self.players_passed_this_round = []
            self.trades_executed_in_current_round = getattr(self, "trades_executed_in_current_round", 0) + 1

        # after each trade, consider bailouts later if needed (handled by maybe_grant_bailout)
        self.enforce_minimum_money_if_needed()

    def start_new_trading_round(self):
        for p in self.players.values():
            p.trades_this_round = 0
        self.players_passed_this_round = []
        self.trades_executed_in_current_round = 0

    def _pay_money(self, agent_name: str, amount: int):
        pl = self.players[agent_name]
        remaining = amount
        to_remove: List[MoneyCard] = []
        pl.money_cards.sort(key=lambda c: c.value, reverse=True)
        for c in pl.money_cards[:]:
            if remaining <= 0:
                break
            if c.value <= remaining and c.value > 0:
                to_remove.append(c)
                remaining -= c.value
        if remaining > 0 and sum(c.value for c in pl.money_cards) >= amount:
            for c in pl.money_cards[:]:
                if c not in to_remove and remaining > 0:
                    to_remove.append(c)
                    remaining -= c.value
        for c in to_remove:
            if c in pl.money_cards:
                pl.money_cards.remove(c)

    def _find_optimal_payment(self, money_cards: List[MoneyCard], target: int) -> List[MoneyCard]:
        n = len(money_cards)
        if n == 0 or target <= 0:
            return []
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        parent = [[None] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            val = money_cards[i - 1].value
            for j in range(target + 1):
                if dp[i - 1][j]:
                    dp[i][j] = True
                    parent[i][j] = (i - 1, j, False)
                if j >= val and dp[i - 1][j - val]:
                    dp[i][j] = True
                    parent[i][j] = (i - 1, j - val, True)
        best = target
        while best >= 0 and not dp[n][best]:
            best -= 1
        if best < 0:
            return []
        res = []
        i, j = n, best
        while i > 0 and j > 0:
            if parent[i][j] is None:
                break
            pi, pj, took = parent[i][j]
            if took:
                res.append(money_cards[i - 1])
            i, j = pi, pj
        return res

    def is_game_over(self) -> bool:
        return self.phase == "game_over"

    def get_winners(self) -> List[str]:
        scores: Dict[str, int] = {}
        for name, p in self.players.items():
            scores[name] = p.calculate_score(self.quartet_values)
        max_score = max(scores.values()) if scores else 0
        return [n for n, s in scores.items() if s == max_score]

    # Minimal enforcement wrapper: used after payments/trades to consider bailout later
    def enforce_minimum_money_if_needed(self):
        """
        Do not provide unconditional repeated stipends. Bailout is only granted via
        maybe_grant_bailout() after multiple forced rounds. This helper exists to keep
        a consistent call site after payments/trades.
        """
        pass


# ----------------- PettingZoo AEC Env wrapper ----------------- #
class KoehandelPettingZooEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "koehandel_v0", "is_parallelizable": False, "render_fps": 2}

    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None, max_turns: Optional[int] = None):
        super().__init__()
        self.num_players = int(num_players)
        self.render_mode = render_mode
        self.max_turns = int(max_turns) if max_turns is not None else None

        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]

        animal_low = [0.0] * 10
        animal_high = [4.0] * 10
        money_low = [0.0] * 6
        money_high = [100.0] * 6
        opponent_low = [0.0] * (self.num_players - 1)
        opponent_high = [5000.0] * (self.num_players - 1)
        game_state_low = [0.0, 0.0, 0.0]
        game_state_high = [40.0, 10000.0, 2.0]

        low_bounds = np.array(animal_low + money_low + opponent_low + game_state_low, dtype=np.float32)
        high_bounds = np.array(animal_high + money_high + opponent_high + game_state_high, dtype=np.float32)

        single_obs_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        self.observation_spaces = {a: single_obs_space for a in self.possible_agents}
        single_action_space = spaces.Discrete(2000)
        self.action_spaces = {a: single_action_space for a in self.possible_agents}

        self.game: Optional[KoehandelGame] = None
        self.agents: List[str] = []
        self._agent_selector = None
        self.agent_selection: Optional[str] = None

        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}

        # used to ensure we can set auctioneer as immediate next actor
        self._skip_next_agent_selector = False

        self.has_reset = False

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observe(self, agent: str) -> np.ndarray:
        return self._get_obs(agent)

    def _get_obs(self, agent_name: str) -> np.ndarray:
        if self.game is None:
            size = 10 + 6 + (self.num_players - 1) + 3
            return np.zeros(size, dtype=np.float32)
        player = self.game.players[agent_name]
        animal_counts = [float(sum(1 for c in player.animal_cards if c.animal_type == t)) for t, _ in self.game.ANIMAL_TYPES]
        money_counts = [float(sum(1 for c in player.money_cards if c.value == v)) for v in self.game.MONEY_VALUES]
        opp_est = [float(player.estimated_opponent_money.get(other, 90.0)) for other in self.game.agent_names if other != agent_name]
        deck_size = float(len(self.game.animal_deck))
        turn_num = float(self.game.turn)
        phase_num = 0.0
        if self.game.phase == "trade":
            phase_num = 1.0
        elif self.game.phase == "game_over":
            phase_num = 2.0
        obs = np.array(animal_counts + money_counts + opp_est + [deck_size, turn_num, phase_num], dtype=np.float32)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.game = KoehandelGame(self.agents, seed=seed)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.game.start_auction()
        self.has_reset = True

    def step(self, action: Optional[int]):
        if not self.has_reset:
            raise RuntimeError("reset before step")
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(self.agent_selection, False):
            return self._was_dead_step(action)
        agent = self.agent_selection
        action_mask = self.game.get_action_mask(agent)
        if action is None:
            action = 0
        if action < 0 or action >= len(action_mask) or action_mask[int(action)] == 0:
            action = 0

        # expose trade options in infos for policies / logging
        trade_options = None
        if self.game is not None and self.game.phase == "trade" and self.game.trade_initiator is None:
            possible_trades = self.game.get_possible_trades(agent)
            trade_options = []
            for idx, (target_name, animal_type, mode) in enumerate(possible_trades):
                trade_options.append({"action": 1000 + idx, "target": target_name, "animal": animal_type, "mode": mode})

        # auctioneer pending info (if any)
        info_dict = {"action_mask": action_mask}
        if trade_options is not None:
            info_dict["trade_options"] = trade_options
        if self.game is not None and self.game.phase == "auctioneer_buy":
            info_dict["pending_auction"] = {
                "pending_winner": self.game.pending_auction_winner,
                "pending_bid": int(self.game.pending_auction_bid),
                "card": str(self.game.current_auction_card) if self.game.current_auction_card is not None else None,
                "auctioneer": self.game.current_auctioneer
            }
        self.infos[agent] = info_dict

        # Apply action to the game
        step_rewards = self.game.apply_action(agent, int(action))
        self.rewards = step_rewards

        # If we entered auctioneer_buy phase, force auctioneer to be next actor
        if self.game.phase == "auctioneer_buy":
            if self.game.current_auctioneer is not None:
                self.agent_selection = self.game.current_auctioneer
                self._skip_next_agent_selector = True

        # Advance turn counter
        self.game.turn += 1

        # Enforce max_turns
        if self.max_turns is not None and self.game.turn >= self.max_turns:
            for a in list(self.agents):
                self.truncations[a] = True
                self.infos[a]["timeout"] = True
            self._handle_game_end()
            return

        # When deck empties, enter forced trading
        if len(self.game.animal_deck) == 0 and self.game.phase == "auction":
            self.game.phase = "trade"
            self.game.forced_trading_mode = True
            self.game.trades_executed_in_current_round = 0
            self.game.players_passed_this_round = []
            self.game.forced_trade_rounds = 0
            self.game.players_passed_trading = []
            self.game.start_new_trading_round()

        # If auction phase and no current card, start auction
        if self.game.phase == "auction" and self.game.current_auction_card is None:
            card = self.game.start_auction()
            if card is None and self.game.phase == "auction":
                self.game.phase = "trade"
                self.game.players_passed_trading = []
                self.game.start_new_trading_round()

        # Accumulate rewards
        for a in self.agents:
            self._cumulative_rewards[a] += self.rewards.get(a, 0.0)

        # Move to next agent
        if not all(self.terminations.values()):
            if self._skip_next_agent_selector:
                self._skip_next_agent_selector = False
            else:
                self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def _handle_game_end(self):
        winners = self.game.get_winners()
        # Normalize to the truly maximum possible final score (total points * number_of_quartets)
        sum_points = sum(points for _, points in KoehandelGame.ANIMAL_TYPES)
        max_quartets = len(KoehandelGame.ANIMAL_TYPES)
        max_possible_score = max(1, sum_points * max_quartets)
        for agent in self.agents:
            player = self.game.players[agent]
            normalized_score = player.score / max_possible_score
            win_bonus = 3.0 if agent in winners else 0.0
            diversity_score = player.get_diversity_score()
            final_reward = normalized_score * 15.0 + win_bonus + diversity_score
            # apply final bailout penalty if player used bailout
            if self.game.bailout_used.get(agent, False):
                final_reward -= BAILOUT_FINAL_PENALTY
            self.rewards[agent] = final_reward
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
        if not all(self.terminations.values()):
            self.agent_selection = self._agent_selector.next()

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\n{'='*60}")
        print(f"Turn {self.game.turn} | Phase: {self.game.phase.upper()}")
        print(f"Deck remaining: {len(self.game.animal_deck)} cards")
        if self.game.phase == "auction" and self.game.current_auction_card:
            print(f"Current auction: {self.game.current_auction_card} (auctioneer: {self.game.current_auctioneer})")
        if self.game.phase == "auctioneer_buy":
            pa = self.game.pending_auction_winner
            pb = self.game.pending_auction_bid
            print(f"Auction finished: pending winner={pa}, bid={pb}; awaiting auctioneer ({self.game.current_auctioneer}) decision")
        if self.game.phase == "trade":
            if self.game.trade_initiator:
                print(f"Trade: {self.game.trade_initiator} vs {self.game.trade_target} (animal: {self.game.trade_animal_type})")
            else:
                print(f"Waiting for trade initiation (passes: {len(self.game.players_passed_trading)}/{len(self.game.agent_names)})")
                if self.game.forced_trading_mode:
                    print(f"Forced-trade round passes: {len(self.game.players_passed_this_round)}/{len(self.game.agent_names)} (rounds={self.game.forced_trade_rounds})")
        print(f"\nPlayer Status:")
        for name, player in self.game.players.items():
            animals = player.get_animal_counts()
            quartets = [a for a, c in animals.items() if c == 4]
            diversity = player.get_diversity_score()
            progress = player.get_progress_score(self.game.quartet_values)
            money = player.total_money()
            bailout_flag = self.game.bailout_used.get(name, False)
            print(f"  {name}: Quartets={quartets}, Progress={progress:.0f}, Diversity={diversity:.1f}, Money={money}, BailoutUsed={bailout_flag}")
        print(f"{'='*60}\n")

    def close(self):
        pass