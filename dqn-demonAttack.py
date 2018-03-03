# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 10000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

ACTION_NOTHING = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3
ACTION_RIGHT_AND_FIRE = 4
ACTION_LEFT_AND_FIRE = 5

if __name__ == "__main__":
    env = gym.make('DemonAttack-v0')
    state_size = np.prod(env.observation_space.shape) #210*__*3
    action_size = env.action_space.n # 6 actions
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset() # shape: (210, 160, 3)
        for time in range(500):
            env.render()
            model_input = state.flatten().reshape([1 ,state_size])
            action = agent.act(model_input)
            next_state, reward, done, _ = env.step(action)
            next_model_input = next_state.flatten().reshape([1 ,state_size])

            reward = reward if not done else -200

            agent.remember(model_input, action, reward, next_model_input, done)

            state = next_state
            print("episode: {}/{}, score: {}, e: {:.2}, reward:{}".format(e, EPISODES, time, agent.epsilon, reward))
            if done:
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        #if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

