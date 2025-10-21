# Koehandel Multi-Agent RL Environment

A PettingZoo-compatible reinforcement learning environment for the card game **Koehandel** (You're Bluffing!), designed for training competitive multi-agent RL policies using RLlib.

## Overview

Koehandel is a bidding and trading card game where players compete to collect complete sets (quartets) of animal cards. This implementation provides a full PettingZoo AEC environment compatible with RLlib's multi-agent training infrastructure.

## Features

- ✅ Full PettingZoo AECEnv API compliance
- ✅ Gymnasium spaces (Box for observations, Discrete for actions)
- ✅ Support for 2-5 players
- ✅ **Flexible bidding**: Combine any money cards to make custom bids
- ✅ Complete game logic: auctions, money management, scoring
- ✅ RLlib training integration with PPO
- ✅ Rendering support for debugging

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pettingzoo>=1.24.0
pip install gymnasium>=0.29.0
pip install ray[rllib]>=2.7.0
pip install torch  # or tensorflow
pip install numpy
```

## Quick Start

### Test the Environment

```python
from koehandel_game_engine import KoehandelPettingZooEnv

# Create environment
env = KoehandelPettingZooEnv(num_players=4, render_mode="human")

# Reset
observations, infos = env.reset(seed=42)

# Game loop
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # Random policy for demonstration
        action = env.action_space.sample()
    
    env.step(action)

env.close()
```

### Train with RLlib

```bash
python train_koehandel.py
```

Training results and checkpoints will be saved to `./results/koehandel_training/`.

## Game Rules

### Setup
- Each player starts with money cards: 2×$0, 4×$10, 1×$50
- Animal deck contains 10 types, 4 cards each (40 total)

### Gameplay

1. **Auction Phase**: Top card is drawn and auctioned
   - Players bid or pass in turn
   - **Players can combine any money cards to bid custom amounts**
   - Highest bidder wins and pays
   - Special: First 4 Donkeys trigger payouts

2. **Trading Phase** (simplified in current implementation)
   - Players can trade animal cards
   - Both players bid secretly; highest wins

3. **Scoring**
   - Complete quartets (4 of same type) score points
   - Values: Horse=1000, Cow=800, Pig=650, etc.
   - Highest score wins

## Environment Details

### Observation Space

`Box(shape=(18,), dtype=float32)`

- **Indices 0-9**: Count of each animal type owned
  - [Horse, Cow, Pig, Donkey, Goat, Sheep, Dog, Cat, Goose, Rooster]
- **Indices 10-15**: Count of each money value owned
  - [$0, $10, $50, $100, $200, $500]
- **Indices 16-17**: Game state
  - Deck size remaining
  - Turn number

### Action Space

`Discrete(1000)`

Players can bid any amount they can afford by combining their money cards.

- **0**: Pass / Don't bid
- **1-999**: Bid the corresponding amount in dollars
  - Example: action=60 means bid $60 (could be $10+$50, or 6×$10, etc.)
  - The game automatically selects the optimal combination of money cards
  - Invalid bids (can't afford) are treated as "pass"

**Examples:**
- Player has cards: [$0, $0, $10, $10, $10, $10, $50]
- Can bid: $10, $20, $30, $40, $50, $60, $70, $80, $90 (and many more combinations)
- Total available: $90

The action space is discrete with 1000 possible actions (0-999), supporting any bid amount up to $999.

### Flexible Bidding System

The environment uses **dynamic programming** to:
1. Calculate all possible bid amounts a player can make
2. Find optimal combinations of money cards for payment
3. Automatically handle card selection when paying

This means agents can bid **any amount** they can afford, not just fixed denominations!

### Rewards

- **During game**: 0 for all agents
- **At game end**:
  - Winners: +1.05
  - Losers: -0.95

## File Structure

```
.
├── koehandel_game_engine.py    # Environment and game logic
├── train_koehandel.py          # RLlib training script
├── README.md                   # This file
└── results/                    # Training outputs (created on run)
```

## Customization

### Change Number of Players

```python
env = KoehandelPettingZooEnv(num_players=3)  # 2-5 players supported
```

### Modify Training Hyperparameters

Edit `train_koehandel.py`:

```python
config = (
    PPOConfig()
    .training(
        lr=1e-4,           # Learning rate
        gamma=0.99,        # Discount factor
        entropy_coeff=0.01,# Exploration
    )
    # ... other settings
)
```

### Extend Action Space

To add trading actions, modify `action_spaces` in `KoehandelPettingZooEnv.__init__`:

```python
# Example: Add trading actions
num_trade_actions = num_players * len(KoehandelGame.ANIMAL_TYPES)
single_action_space = spaces.Discrete(1000 + num_trade_actions)
```

## Technical Details

### Money Card Combination Algorithm

The environment uses a **subset sum dynamic programming** approach:

1. **Finding possible bids**: Computes all possible sums from available money cards
2. **Optimal payment**: Finds the best combination of cards to pay exact amounts
3. **Time complexity**: O(n × target) where n is number of cards

This allows agents to explore the full strategy space of bidding!

## Known Limitations

- Trading phase is simplified (not fully implemented)
- Action masking not yet implemented (agents can try invalid actions)
- No support for "matching bid" by auctioneer

## Roadmap

- [ ] Full trading/Koehandel mechanic
- [ ] Action masking for invalid moves
- [ ] Self-play training configuration
- [ ] Pretrained baseline agents
- [ ] Web-based visualization

## References

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [RLlib Multi-Agent](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-environments)
- [Koehandel Rules](https://boardgamegeek.com/boardgame/6794/youre-bluffing)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Happy Training! 🎲🐄🐴**

Created by: @FrozenSmoothie
Date: 2025-10-21