<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/August/5b81cd05_soccer/soccer.png">

# DRL - Multi-Agent A2C with PPO - Soccer

## Overview
Using the Unity "Soccer" environment, this deep reinforcement learning task trains a two player AI team to play soccer against an opposing "random team." Each team consists of a goalie and striker that are rewarded for each goal made against the opposing team and penalized when a goal is scored against them. The task is considered solved when the trained AI team consistently beats the random team. The random team is untrained and only takes random actions on the field.

## A2C with PPO
As seen in the code, the Advantage Actor Critic (A2C) framework is used to train the two agents (goalie and striker.) It employs Proximal Policy Optimization (PPO) to optimize learning. Policy gradient-based methods like A2C with PPO are very effective in environments where there are a limited set of discrete actions for an agent to take at each step. 

In the Soccer environment, each player can choose only 4 (goalie) or 6 (striker) different actions, such as forward, backwards, left, right and spin. Each agent "sees" a 180 degree view in front of them, including the various objects on the field and their relative placement and distance from the agent.

My implementation of Soccer uses two deep neural networks (an actor and critic) for each agent (goalie and striker) that over time learns the most rewarding action to choose in a given state. The agents play one round of soccer, collecting all of the experiences in that round, then use them to learn better actions for the next round. I have used PPO with a clipped surrogate function to limit how much the agents update their learning after each successive game. This avoids agents gathering a set of experiences in one round of play that "appear" to contain very effective actions and going too far in that direction, which risks them becoming stuck in a non-optimal long term pattern of play. This limiting (clipping) function is at the heart of the PPO algorithm and its effectiveness.

## Results
After approximately 25,000 games played, equating to several million time steps, the trained team beat the random team in 100 consecutive episodes. This demonstrates near flawless performance by both the stiker, who had to score every game for 100 games in a row without a miss, and the goalie, who had to prevent the random team from scoring even a single goal over 100 games. While it may seem a team taking random actions is easy to beat, it is not. It's actually quite surprising how many goals a random team can make, as seen in the early episodes of training where the score is -1.

## Setup Instructions

To reproduce this model on a Mac:

1. Install the <a href="https://www.anaconda.com/download/#macos">Anaconda distribution of Python 3</a>

2. Install PyTorch, Jupyter Notebook and Numpy in the Python3 environment.

3. Clone the <a href="https://github.com/udacity/deep-reinforcement-learning">Udacity DRLND repo</a> and install the dependencies by typing the following:

    git clone https://github.com/udacity/deep-reinforcement-learning.git

    cd deep-reinforcement-learning/python

    pip install .
    
4. Download the <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip">custom Unity Soccer agent for Mac OSX</a> and save it in the p3_collab_compet folder of the Udacity repo.

5. Open Jupyter Notebook and run the Soccer-PPO.ipynb file to train the agent.

6. To watch the agents I trained play tennis, copy the 2 checkpoint files into the p3_collab_compet folder and execute all the notebook cells except sections 8 and 9 (the two sections above "Watch The Trained Agent.")
