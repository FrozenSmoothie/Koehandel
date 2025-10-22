"""
Koehandel game engine (PettingZoo-style AEC environment)

This file provides:
- KoehandelPettingZooEnv: a PettingZoo-compatible AEC environment for the
  Koehandel training script. It's a simplified but fully working environment
  that exposes observation_space, action_space, possible_agents, reset, step,
  observe, and close.
- Scoring helpers implementing your requested scoring rule:
    final_score = total_points * number_of_quartets
  where total_points = sum(points_per_quartet * quartets_of_that_type)

Notes:
- The environment is intentionally simplified to be stable for RL training while
  preserving the core "collect quartets and score" mechanic. Replace or extend
  the logic where noted if you want full game rules (trading/auction/etc).
- Observations are returned as a fixed-length numpy.float32 vector (shape (22,))
  to match the shape expected by the rest of your training code.
- Action space is Discrete(2000) (keeps compatibility with your previous runs).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from gym import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import random

# Values per completed quartet (adjust if your game has different animals)
DEFAULT_QUARTET_VALUES: Dict[str, int] = {
    "horse": 1000,
    "cow": 800,
    "pig": 650,
    "sheep": 500,
    "hen": 300,
}


def compute_quartets_from_inventory(inventory: Dict[str, int]) -> Dict[str, int]:
    """
    Convert raw animal counts to number of complete quartets per animal type.
    """
    quartets: Dict[str, int] = {}
    for animal, count in inventory.items():
        try:
            cnt = int(count)
        except Exception:
            cnt = 0
        quartets[animal] = max(0, cnt // 4)
    return quartets


def compute_score_from_quartets(quartets: Dict[str, int], values: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Apply scoring rule:
      total_points = sum(points_per_quartet * quartets_of_that_type)
      number_of_quartets = sum(quartets)
      final_score = total_points * number_of_quartets
    Returns a dict with score, total_points, number_of_quartets, breakdown.
    """
    if values is None:
        values = DEFAULT_QUARTET_VALUES

    total_points = 0
    total_quartets = 0
    for animal, qcount in quartets.items():
        pts = values.get(animal, 0)
        total_points += pts * qcount
        total_quartets += qcount

    final_score = int(total_points * total_quartets) if total_quartets > 0 else 0
    return {
        "score": final_score,
        "total_points": int(total_points),
        "number_of_quartets": int(total_quartets),
        "breakdown": dict(quartets),
    }


def compute_score_from_inventory(inventory: Dict[str, int], values: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    quartets = compute_quartets_from_inventory(inventory)
    return compute_score_from_quartets(quartets, values=values)


# -------------------------
# PettingZoo-style environment
# -------------------------
class KoehandelPettingZooEnv(AECEnv):
    """
    Simplified Koehandel environment. Multi-agent turn-taking.

    Config options (passed as env_config dict to the constructor):
      - num_players: number of players (default 4)
      - max_turns: maximum number of turns (global), None means a default cap (300)
    """

    animals: List[str] = ["horse", "cow", "pig", "sheep", "hen"]

    def __init__(self, num_players: int = 4, max_turns: Optional[int] = None):
        super().__init__()
        assert num_players >= 2, "Need at least 2 players"
        self.num_players = num_players
        self.possible_agents: List[str] = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]  # active agents list for pettingzoo API
        self.max_turns = max_turns or 300
        self.current_agent: Optional[str] = None
        # ensure agent_selector is callable whether it was imported as a module or a function
        if hasattr(agent_selector, 'agent_selector'):
            _agent_selector_callable = agent_selector.agent_selector
        else:
            _agent_selector_callable = agent_selector
        self._agent_selector = _agent_selector_callable(self.agents)
        # observation and action spaces (kept stable and compatible with training script)
        # Observation: 22-d vector of floats (matches your earlier observed shape)
        # layout (example):
        #  - animals owned counts: len(animals) -> 5
        #  - money (1)
        #  - current_turn (1)
        #  - last_offer (5) - example features
        #  - padding -> fill to 22
        obs_low = np.zeros(22, dtype=np.float32)
        obs_high = np.array([1e4] * 22, dtype=np.float32)
        self._observation_spaces = {agent: spaces.Box(low=obs_low, high=obs_high, dtype=np.float32) for agent in self.possible_agents}

        # Keep action space as Discrete(2000) for compatibility with your previous code.
        self._action_spaces = {agent: spaces.Discrete(2000) for agent in self.possible_agents}

        self.seed()
        self.reset()

    # PettingZoo API property style spaces
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    # Standard RNG seed
    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.randrange(2 ** 30)
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # reset environment state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Per-player inventories (animal counts)
        self.inventories: Dict[str, Dict[str, int]] = {
            agent: {animal: 0 for animal in self.animals} for agent in self.agents
        }
        # initial money for each player
        self.money: Dict[str, int] = {agent: 100 for agent in self.agents}

        # bookkeeping
        self.turn_count = 0
        self.steps = 0
        self.current_agent = self._agent_selector.next()
        # last action/offer representation (for observation)
        self.last_offer = {agent: 0 for agent in self.agents}

        # initial distribution: give each agent some starting animals to let quartets form over time
        for agent in self.agents:
            for animal in self.animals:
                self.inventories[agent][animal] = random.randint(0, 3)
        # Guarantee that episodes are not degenerate
        self._cumulative_steps = 0

    def observe(self, agent: str) -> np.ndarray:
        """
        Return a fixed-size observation for the given agent.
        Observation layout (example):
          [counts(5), money(1), current_turn(1), last_offer(5), padding ...] -> length 22
        """
        inv = self.inventories[agent]
        counts = np.array([inv[a] for a in self.animals], dtype=np.float32)
        money = np.array([self.money[agent]], dtype=np.float32)
        current_turn = np.array([float(self.turn_count)], dtype=np.float32)
        last = np.array([self.last_offer.get(agent, 0) for _ in range(len(self.animals))], dtype=np.float32)
        # pad to 22
        obs = np.concatenate([counts, money, current_turn, last])
        if obs.size < 22:
            pad = np.zeros(22 - obs.size, dtype=np.float32)
            obs = np.concatenate([obs, pad])
        elif obs.size > 22:
            obs = obs[:22]
        return obs

    def _collect_reward_if_completed(self, agent: str) -> float:
        """
        Internal helper: compute score now (not only at end). For simplicity, returns 0
        per intermediate step. Final reward is computed on episode end.
        """
        return 0.0

    def step(self, action: int):
        """
        Process an action from the current agent. This simplified environment
        treats actions as one of:
          - small integer space mapped into some simple behaviors:
            * 0..9 : pass / do nothing
            * 10..999 : attempt to acquire a random animal (spend money)
            * 1000..1999 : play an 'auction' that yields random animals to winner
        The aim is to produce varied inventories so PPO can learn to collect quartets.
        """
        if self.dones.get(self.current_agent, False):
            # if the current_agent is done, advance to next
            self.current_agent = self._agent_selector.next()
            return

        agent = self.current_agent
        # default small reward
        reward = 0.0

        # Simple mapping of action ranges to behaviors (you can override with real game logic)
        if 0 <= action <= 9:
            # pass/do nothing
            pass
        elif 10 <= action < 1000:
            # "buy" an animal at a fixed (small) cost. Map action to animal index
            animal_idx = (action - 10) % len(self.animals)
            animal = self.animals[animal_idx]
            cost = 5
            if self.money[agent] >= cost:
                self.money[agent] -= cost
                self.inventories[agent][animal] += 1
                self.last_offer[agent] = animal_idx
        elif 1000 <= action < 2000:
            # "auction" simulation: 10% chance to win a random animal with some cost
            if random.random() < 0.1:
                animal = random.choice(self.animals)
                self.inventories[agent][animal] += 1
        else:
            # out-of-range actions are treated as pass
            pass

        # Occasionally give a free animal to keep training moving
        if random.random() < 0.05:
            animal = random.choice(self.animals)
            self.inventories[agent][animal] += 1

        # bookkeeping
        self.turn_count += 1
        self.steps += 1
        self._cumulative_steps += 1

        # collect immediate reward (if any)
        r = self._collect_reward_if_completed(agent)
        self.rewards[agent] = r
        self._cumulative_rewards[agent] += r

        # termination condition
        # End when cumulative steps exceeds max_turns or when any player reaches some quartets threshold
        done_now = False
        if self._cumulative_steps >= self.max_turns:
            done_now = True
        else:
            # if any player completes >= 5 quartets total, end early for speed
            for a in self.agents:
                q = sum(compute_quartets_from_inventory(self.inventories[a]).values())
                if q >= 5:
                    done_now = True
                    break

        if done_now:
            # compute final scores and assign terminal rewards
            scores = {}
            for a in self.agents:
                sc = compute_score_from_inventory(self.inventories[a], values=DEFAULT_QUARTET_VALUES)
                scores[a] = sc["score"]
            # Provide final reward as the raw score (float) for each agent
            for a in self.agents:
                self.rewards[a] = float(scores[a])
                self._cumulative_rewards[a] = self.rewards[a]
            # mark done for all agents
            for a in self.agents:
                self.dones[a] = True
            self.terminations = {a: True for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            # Advance selector so observe/step semantics don't break
            self.current_agent = None
            return

        # otherwise, advance to next agent for the next call
        self.current_agent = self._agent_selector.next()

    def render(self, mode="human"):
        """
        Simple textual render. Useful for debugging.
        """
        lines = [f"Turn: {self.turn_count} / Steps: {self._cumulative_steps}"]
        for a in self.agents:
            inv = self.inventories[a]
            q = compute_quartets_from_inventory(inv)
            sc = compute_score_from_quartets(q)
            lines.append(f"{a}: inv={inv} quartets={q} score={sc['score']}")
        out = "\n".join(lines)
        if mode == "human":
            print(out)
        return out

    def close(self):
        # Nothing special to close in this simplified env
        pass