A repository created while learning the fundamentals of multi-armed bandits.

Contains:
- `bandit_dgp`: Which simulates outcomes from pulls of multi-armed bandits (Bernoulli trials)
- `epsilon_greed`: Which has the following algorithms for exploration-exploitation:
  - greedy (can have optimistic initial values)
  - epsilon-greedy (can include a decay rate for epsilon)
- `rl.ipynb`: Which contains basic examples and code for reinforcement learning (from Hands-on Machine Learning by Aurelien)including:
  - Basic policy gradient
  - Q-values conceptualisation and formulation
  - Deep Q-learning