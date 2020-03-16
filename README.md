# udacity_rl_project_two
udacity reinforcement learning project 2: continuous control

## The Environment

For this project, student have to train an agent (20 agents) represented by a double-jointed arm which moves to target locations. A **reward** of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The **state space** consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

Each **action** is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

The task is episodic, and in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes.


## Actor & Critic Networks

To solve the problem multilayer (2 hidden layers) dense neural networks were trained:

* Actor Networks: *input* - state vector, *hidden* - two layers 80 neurons, *output* - action 4d vector;
* Critic Network: *input* - state & action vectors, *hidden* - two layers 24 & 48 neuron, *output* - advantage scalar value.

As part of the project, the task was solved using the following algorithms:

* DDPG - Deep Deterministic Policy Gradient
* PPO - Proximal Policy Optimization

