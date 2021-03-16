#  A car is on a one-dimensional track, positioned between two "mountains".
#  The goal is to drive up the mountain on the right; however,
#  the car's engine is not strong enough to scale the mountain in a single pass. Therefore,
#  the only way to succeed is to drive back and forth to build up momentum.
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import gym
import random
import math
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import numpy as np

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make('MountainCar-v0')
env.seed(110)
np.random.seed(10)
VIDEO_PATH = "exp1.mp4"

class DQN:
    """ Implementation of deep q learning algorithm """
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .95
        self.batch_size = 64 
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .995
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=self.state_space, activation=relu))
        model.add(Dense(4, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch   = random.sample(self.memory, self.batch_size)
        states      = np.array([i[0] for i in minibatch])
        actions     = np.array([i[1] for i in minibatch])
        rewards     = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones       = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_reward(state):
    if state[0] >= 0.5:
        print("Car has reached the goal")
        return 100
    if state[0] > -0.4:
        return (1+state[0])**2
    return 0

def train_dqn(episode):
    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    video_recorder = None
    video_recorder = VideoRecorder(env, VIDEO_PATH, enabled=True)
    
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        max_steps = 250
        for i in range(max_steps):
            env.unwrapped.render()
            video_recorder.capture_frame()
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, 2))
            # reward = get_reward(next_state)
            reward = 1000*((math.sin(3*next_state[0,0]) * 0.0025 + 0.5 * next_state[0,1] * next_state[0,1]) - (math.sin(3*state[0,0]) * 0.0025 + 0.5 * state[0,1] * state[0,1])) 
            reward -= 0.001
            if state[0,0] >= 0.5:
                reward +=5
            score += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        
    agent.model.save('mountain_car_84_500.hdf5')
    return loss

if __name__ == '__main__':
    episodes = 5
    loss = train_dqn(episodes)
    plt.plot([i+1 for i in range(episodes)], loss)
    plt.savefig("exp1.png")
    env.close()