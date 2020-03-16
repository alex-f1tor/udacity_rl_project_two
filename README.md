# udacity_rl_project_two
udacity reinforcement learning project 2: continuous control

## The Environment

For this project, student have to train an agent (20 agents) represented by a double-jointed arm which moves to target locations.

![Image](https://github.com/alex-f1tor/udacity_rl_project_two/blob/master/imgs/Continuous%20Control%20DEMO.png)

A **reward** of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The **state space** consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

Each **action** is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

The task is episodic, and in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes.

---

The following python3 libraries are required:

`numpy == 1.16.2`

`pytorch == 0.4.0` - (GPU enabled)

`unity ML-agent` - available at [github](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

---

## Actor & Critic Networks

To solve the problem multilayer (2 hidden layers) dense neural networks were trained:

* Actor Networks: *input* - state vector, *hidden* - two layers 80 neurons, *output* - action 4d vector;
* Critic Network: *input* - state & action vectors, *hidden* - two layers 24 & 48 neuron, *output* - advantage scalar value.

As part of the project, the task was solved using the following algorithms:

* DDPG - Deep Deterministic Policy Gradient [notebook](https://github.com/alex-f1tor/udacity_rl_project_two/blob/master/DDPG/Continuous_Control.ipynb)

* PPO - Proximal Policy Optimization [notebook](https://github.com/alex-f1tor/udacity_rl_project_two/blob/master/PPO/Continuous_Control.ipynb)


## Results

Surprisingly, DDPG allows to get more stable and faster solution than PPO. Moreover, with a smaller value of discount factor (*gamma*), faster convergence is observed:

![Image](https://github.com/alex-f1tor/udacity_rl_project_two/blob/master/imgs/ddpg_gamma.png)

The PPO algorithm demonstrated a less stable training process: often the average score reached a value of >3 and then started to fall to value 0.2. Note that, unlike DDPG, the memory buffer should be cleared each episode, therefore during the training only action-reward pairs that appeared in the current episode are used. Unfortunately for current task this approach didn't lead to the desired result, therefore, buffer clearing was introduced every 10 episodes (buffer size 10** 4). This approach provided model training, although it was still slower than for DDPG:

![Image](https://github.com/alex-f1tor/udacity_rl_project_two/blob/master/imgs/ppo_gamma.png)




