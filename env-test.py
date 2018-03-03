'''
A file for getting familiar with the Gym environment.
Use a debugger to inspect various parts of code for better understanding.
'''

import gym

env = gym.make('DemonAttack-v0')
EPISODES = 1000
ACTION_NOTHING = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3
ACTION_RIGHT_AND_FIRE = 4
ACTION_LEFT_AND_FIRE = 5

state = env.reset()
for e in range(EPISODES):
     state = env.step(ACTION_FIRE)
     env.render()
     print(e, state[1])