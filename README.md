<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/August/5b81cd05_soccer/soccer.png">

# DRL - Multi-Agent PPO - Soccer

## Overview
Using the Unity agent/environment "Soccer", this deep reinforcement learning task trains a two player AI team to play soccer against an opposing "random team." Each team consists of a goalie and striker that are rewarded for each goal made against the opposing team and penalized when a goal is scored against them. The task is considered solved when the trained AI team consistently beats the random team. The random team is untrained and only takes random actions on the field.

## PPO Algorithm
As seen in the code, a policy gradient-based method is used to train the two agents (goalie and striker.) It employs the Proximal Policy Optimization (PPO) algorithm to optimize learning. Policy gradient-based methods with PPO are very effective in environments where there are a limited set of actions for an agent to take at each step. 

In the Soccer environment, each player can choose only 4 (goalie) or 6 (striker) different movements, such as forward, backwards, left, right and spin. My implementation for Soccer uses one deep neural network for each agent (goalie and striker) that over time learns the highest probability action to choose in a given state. The agents play one round of soccer, collecting all of the experiences in that round, then use them to learn better actions for the next round. I have used PPO with a clipped surrogate function to limit how much the agents update their learning after each successive game. This avoids agents gathering a set of experiences in one round of play that "appear" to contain very effective actions and going too far in that direction, which risks them becoming stuck in a non-optimal long term pattern of play. This limiting (clipping) function is at the heart of the PPO algorithm and its effectiveness.

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
