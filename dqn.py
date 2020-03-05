import numpy as np
import random

from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

np.random.seed(1)
random.seed(1)

class DQNAgent:

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.batch_size = 25

        self.alpha = 0.01
        self.epsilon = 0.5
        self.epsilon_decay = 0.99
        self.gamma = 0.9

        self.memory = deque(maxlen=20000)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_dim=self.num_states))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        return model

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)

        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def memoize(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, new_state, done in minibatch:
            q_update = reward if done else (reward + self.gamma * (self.model.predict(new_state)[0]).max())
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

        self.epsilon *= self.epsilon_decay


