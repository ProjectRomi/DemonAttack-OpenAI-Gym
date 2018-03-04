# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 10000


import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.models import Model
from timeit import default_timer as timer

WIDTH = 84
HEIGHT = 84
CHANNELS = 3
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.97 #0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        with tf.device("/cpu:0"):
            state = tf.placeholder("float", [None, CHANNELS, WIDTH, HEIGHT])
            inputs = Input(shape=(CHANNELS, WIDTH, HEIGHT,))
            model = Conv2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4, 4), activation='relu',
                                  border_mode='same')(inputs)
            model = Conv2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu',
                                  border_mode='same')(model)
            model = Flatten()(model)
            model = Dense(output_dim=256, activation='relu')(model)
            q_values = Dense(output_dim=action_size, activation='linear')(model)
            m = Model(input=inputs, output=q_values)

            m.compile(loss='mse',
                          optimizer=Adam(lr=self.learning_rate))
        return m

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        print("memory size:", len(self.memory))

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


from skimage.transform import resize
from skimage.color import rgb2gray

prev_frame_0 = np.zeros((WIDTH, HEIGHT))
prev_frame_1 = np.zeros((WIDTH, HEIGHT))
prev_frame_2 = np.zeros((WIDTH, HEIGHT))

def get_preprocessed_frame(observation):
    """
    See Methods->Preprocessing in Mnih et al.
    1) Get image grayscale
    2) Rescale image
    """
    global prev_frame_0
    global prev_frame_1
    global prev_frame_2

    prev_frame_0 = prev_frame_1[:]
    prev_frame_1 = prev_frame_2
    prev_frame_2 = resize(rgb2gray(observation), (WIDTH, HEIGHT))
    return np.concatenate([prev_frame_0, prev_frame_1, prev_frame_2]).reshape([1, CHANNELS, WIDTH, HEIGHT])

if __name__ == "__main__":
    env = gym.make('DemonAttack-v0')
    state_size = np.prod(env.observation_space.shape) #210*160*3
    action_size = env.action_space.n # 6 actions
    agent = DQNAgent(state_size, action_size)

    agent.load("./save/demondattack-dqn-conv.h5")

    done = False
    batch_size = 112

    for e in range(EPISODES):
        state = env.reset() # shape: (210, 160, 3)
        lives = 0;
        last_updated = 0;
        for time in range(1000):
            env.render()
            model_input = get_preprocessed_frame(state)
            action = agent.act(model_input)
            next_state, reward, done, info = env.step(action)
            next_model_input = get_preprocessed_frame(next_state)

            reward = -50 if reward == 0 else reward
            reward = -20 if lives > info['ale.lives'] else reward
            reward = reward if not done else -80
            lives = info['ale.lives']

            agent.remember(model_input, action, reward, next_model_input, done)

            state = next_state
            print("episode: {}/{}, score: {}, e: {:.2}, reward:{}, life:{}, epsilon: {}".format(e, EPISODES, time, agent.epsilon, reward, info, agent.epsilon * 100))
            if done:
                break
        if len(agent.memory) > batch_size:
            start = timer()
            agent.replay(batch_size)
            print(timer() - start)
            agent.save("./save/demondattack-dqn-conv.h5")

