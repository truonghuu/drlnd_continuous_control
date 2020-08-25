# Deep Reinforcement Learning Nanodgree 
## Project on Continuous Control

### Project Description

In this project, we train an agent to solve the Reacher Unity environment, in which a double-jointed arm can move to target location. A reward of +0.1 is provided for each step if the agent's hand is in the goal location. Thus, to maximize the reward, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The environment is solved when the agent achieves an average reward of +30 over 100 consecutive episodes. 

In this project, we use an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm to train the agent.

### Environment Description

The state space of the environment consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Learning Algorithm

Since the action of the agent is continuous, we adopt Deep Deterministic Policy Gradient (DDPG) algorithm, which is a model-free off-policy algorithm to train the agent. DDPG combines the ideas of DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network) in which Actor and Critic, each having two neural networks: a regular network and a target network. The target networks are actually the copies of the regular networks but slowly learns, thus improving the stability in training. In this algorithm, The Actor network is used to determine the action given a state while the Critic network is used to estimate the Q-value for a pair of (`state, action`). We note that unlike the Advantage Actor-Critic, DDPG directly maps a `state` to the best `action` rather than providing the probability distribution across a discrete action space. This reduces the effort to discretesize the action space, which in this case is continuous. The below figure presents the pseudo-codes of the DDPG algorithm (Image taken from “Continuous Control With Deep Reinforcement Learning” (Lillicrap et al, 2015)).

![DDPG Algorithm](figures/1*BVST6rlxL2csw3vxpeBS8Q.png)

#### Experience Replay

Since DDPG adopts DQN, it also uses a replay buffer, which is a finite-sized memory (we have set the size of the reppay buffer to 100000) to store all the experience tuples (state, action, reward, next_state). At every time step, the algorithm randomly samples a mini-batch from the replay buffer to update the value and policy networks. In the code, we have set the mini-batch size to 128. It is worth recalling that experience replay helps to break the temporal/chronological correlation among state/action pairs in each training episode. Withouth experience replay, this correlation could lead to instability (oscillation or divergence of Q-Values) during training as small updates to Q-values may significantly change the policy.
