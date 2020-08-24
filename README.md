# Deep Reinforcement Learning Nanodgree 
## Project on Continuous Control

### Project Description

In this project, we train an agent to solve the Reacher Unity environment, in which a double-jointed arm can move to target location. A reward of +0.1 is provided for each step if the agent's hand is in the goal location. Thus, to maximize the reward, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The environment is solved when the agent achieves an average reward of +30 over 100 consecutive episodes. 

In this project, we use an actor-critic algorithm, the Deep Deterministic Policy Gradients (DDPG) algorithm to train the agent.

### Environment Description

The state space of the environment consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Code Overview

The repository consists of several files:

Continuous_Control.ipynb - the main notebook which will be used to run and train the agent.
agent.py - defines the Agent that is being trained
model.py - defines the PyTorch model for the Actor and the Critic network
checkpoint_actor.pth - stores the weights of the trained Actor network when the environment is solved 
checkpoint_critic.pth - stores the weights of the trained Critic network when the environment is solved 

### Getting Start

To be able to run this project, you need to install dependencies by following the instructions in this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

A prebuilt environment has to be installed. The code developed in this project uses only one (1) agent. Thus, you need to download the corresponding environment depending on your operating system.

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Let us start! Open `Continuous_Control.ipynb` and follow step by step to train the agent and see its working.
