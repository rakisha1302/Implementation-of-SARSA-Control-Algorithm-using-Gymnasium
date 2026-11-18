# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the SARSA (State-Action-Reward-State-Action) control algorithm using the Gymnasium FrozenLake environment. The agent learns an optimal action-selection policy through trial and error by updating its Q-table based on rewards and the next action selected. The objective is to reach the goal while avoiding holes.


## Software Requirements

Requirement	Description
Programming Language	Python
Platform	Jupyter Notebook
Library 1	Gymnasium
Library 2	NumPy
Library 3	Matplotlib
Environment	FrozenLake-v1
Algorithm	SARSA (On-Policy TD Learning)

## Environment Description

FrozenLake is a grid-based environment provided by Gymnasium.
- The environment consists of a 4 × 4 grid.
- The agent starts at the starting state (S).
- The goal is to reach the destination (G).
- The agent must avoid holes (H).
- Frozen surfaces (F) allow the agent to move.
- Four actions are available: Left, Down, Right and Up.
- The environment is slippery, so the agent may move in an unintended direction.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
1. Start the FrozenLake environment using Gymnasium.
2. Initialize the Q-table with zeros.
3. Set the hyperparameters: learning rate, discount factor and exploration rate.
4. Select an action using the epsilon-greedy policy.
5. Perform the selected action and observe the reward and next state.
6. Select the next action using the epsilon-greedy policy.
7. Update the Q-value using the SARSA update formula.
8. Move to the next state and repeat until the episode terminates or the maximum number of steps is reached.
9. Store the episode reward and decrease epsilon using the decay factor.
10. Repeat the training process for 10,000 episodes.
11. Calculate the state-value function and learned policy.
12. Display the Q-table, learned policy, average reward and learning curve.


## Python Program

```python


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create FrozenLake Environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Hyperparameters
num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Epsilon-Greedy Action Selection
def epsilon_greedy_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

# SARSA Training
episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()
    action = epsilon_greedy_action(state, epsilon)
    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        next_action = epsilon_greedy_action(next_state, epsilon)

        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Display Functions
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# Output
print("\nName: Rakisha R")
print("\nRef no: 212225230223")
print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# Plot Learning Curve
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()

```
---

## Output

<img width="647" height="772" alt="image" src="https://github.com/user-attachments/assets/a17fec58-4469-44f5-9931-094f29d54f51" />

<img width="1010" height="597" alt="image" src="https://github.com/user-attachments/assets/e1154c28-7b72-42c5-a5a8-406f81acbc8f" />

---

## Result
```text

Thus, the SARSA control algorithm was successfully implemented
using the Gymnasium FrozenLake-v1 environment.

The agent learned the action-value function through repeated
interaction with the environment and updated the Q-table
using the SARSA update rule.

The learned policy, state-value function and learning curve
were obtained successfully.

```

---

## Inference
```text
The SARSA algorithm learns an action-selection policy through
trial and error using the On-Policy Temporal Difference
learning technique.

The epsilon-greedy strategy balances exploration and
exploitation. The epsilon value gradually decreases during
training, allowing the agent to shift from random exploration
towards selecting better-known actions.

The Q-table stores the estimated values of different actions
in each state. The state-value function identifies the maximum
Q-value for each state, while the learned policy selects the
action with the highest Q-value.

The learning curve represents the average reward over
500 episodes. A higher average reward indicates that the
agent is reaching the goal more frequently.


```
---

