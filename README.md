# BasketWorld

BasketWorld is a grid-based, Gym-compatible simulation of 3-on-3 half-court basketball. It is designed for reinforcement learning research into emergent coordination, strategy, and multi-agent decision-making using a shared policy framework.

---

## 🧠 Core Concepts

- **Grid-Based Court**: A 12x8 half-court with discrete agent movements and a hoop on one side.
- **3v3 Agents**: Each team has 3 players. One shared policy controls all agents.
- **Simultaneous Actions**: All 6 agents act at each timestep.
- **Role-Conditioned Learning**: Observations and rewards are tailored to each agent's role (offense/defense).
- **Gym-Compatible**: Standard `reset()`, `step()`, and `render()` APIs.

---

## 📂 Project Structure

```
basketworld/
├── basketworld/
│   ├── envs/               # Environment and wrappers
│   ├── models/             # Shared PyTorch policy networks
│   ├── sim/                # Core simulation logic (court, rules, game state)
│   └── utils/              # Rendering, reward helpers, etc.
├── train/                  # PPO training scripts and configs
├── tests/                  # Unit tests
├── assets/                 # Logos and visual assets
├── notebooks/              # Exploration and analysis notebooks
├── scripts/                # Dataset or rollout tools
├── README.md
├── setup.py
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run a quick test
python train/train_ppo.py
```

---

## 🤖 Reinforcement Learning Ready

BasketWorld is designed to work with single-policy RL algorithms like PPO, A2C, or DQN by:
- Exposing individual agent observations in a consistent format
- Using role-specific reward shaping
- Providing full control over environment dynamics and rendering

---

## 🏀 Use Cases
- Emergent passing and team play
- Defensive strategy learning
- Curriculum training from 1v1 to 3v3
- Multi-agent transfer learning
- Simulation-based basketball strategy research

---

## 📜 License
MIT License

---

## 🧩 Credits
BasketWorld is inspired by classic RL environments like [GridWorld](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) and adapted for multi-agent, role-based learning in sports simulation.

> For ideas, bugs, or contributions — open an issue or pull request!
