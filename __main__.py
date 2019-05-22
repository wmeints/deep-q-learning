import numpy as np
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count

from policy import DQN
from torchagent.agents import DQNAgent
from torchagent.memory import SequentialMemory
from torchagent.policy import DecayingEpsilonGreedyPolicy

from environment import create_environment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = create_environment()
model = DQN(80,88,1, env.action_space.n).to(device)

loss = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
memory = SequentialMemory(1000000)

# Define an agent that uses the experience buffer and value-function model that we created before.
# The agent will try to optimize the value-function to estimate the value of each possible state/action pair.
#
# The agent uses a decaying epsilon-greedy policy to determine the action to execute in the environment.
# We're moving from 100% random to 0.1% chance of random actions in 1M steps. 
# This means, we'll be totally random at the start and slowly moving towards a deterministic policy.
# 
# Note on the gamma parameter: this controls the discount of future rewards when it comes to estimating
# the value-function. A reward received from the current action accounts for 100% of the value of an action
# in the current state. Any possible action in the next state only contributes 99% of its reward towards the
# value in the current state!
# 
# Note on the tau parameter: this controls the speed at which we update the value-function network in the agent.
# We're using a pair of neural networks in the agent to help with the optimization process for the value-function.
# Setting a tau value larger than 1 means that the target network only gets updated after x steps. A value of less
# then 1 means that we're gradually updating the target network.
# The target network construct is important, because it allows us to more predictably learn the value-function.
agent = DQNAgent(env.action_space.n, model, loss, 
    optimizer, memory, tau=10000, gamma=0.99, warmup_steps=50000, update_steps=3,
    policy=DecayingEpsilonGreedyPolicy(env.action_space.n, 1.0, 1000000, 0.1, 1.0))

episodes = []

# We're going to play 800 games of space invaders.
# This will take a very long time on your CPU, use at your own risk!
for i in range(800):
    state = env.reset()
    episode_reward = 0

    env.render()

    # Run the episode in terms of timesteps (t)
    # This is where the bot actually plays the game
    for t in count():
        # Choose an action and perform it on the environment.
        # The output is a new state, the reward and termination condition.
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        reward = torch.tensor([reward], device=device)

        # Record the experience with the agent and train the agent.
        # This performs a single backward step through the neural network
        # using a number of previous state/action combinations to find
        # a better value-estimation function.
        agent.record(state, action, next_state, reward, done)
        agent.train()

        env.render()

        # Store the new state as the current state 
        # and move on to the next step if we haven't reached the termination point.
        state = next_state

        if done:
            print('Episode %i finished in %i timesteps with reward %f' % (i, t, episode_reward))
            break
