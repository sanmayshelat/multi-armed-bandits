A repository created while learning the fundamentals of multi-armed bandits.

Contains:
- `bandit_dgp`: Which simulates outcomes from pulls of multi-armed bandits (Bernoulli trials)
- `mab_algo`: Which has the following algorithms for exploration-exploitation:
  - greedy (can have optimistic initial values)
  - epsilon-greedy (can include a decay rate for epsilon)
  - ucb1 (optimism in the face of uncertainty)
  - thompson-sampling (probability matching/Bayesian)
- `rl.ipynb`: Which contains basic examples and code for reinforcement learning (from Hands-on Machine Learning by Aurelien)including:
  - Basic policy gradient
  - Q-values conceptualisation and formulation
  - Deep Q-learning
- `rl.py`: Which contains a class-based approach to solving the basic CartPole problem