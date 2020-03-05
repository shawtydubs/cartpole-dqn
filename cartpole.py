import gym
import numpy as np
import time
from collections import deque

from dqn import DQNAgent

ENV = 'CartPole-v0'
WINNING_SCORE = 195
MAX_STEPS = 200

ENV = 'CartPole-v1'
WINNING_SCORE = 475
MAX_STEPS = 500

EPOCHS = 1000
NUM_SCORES = 100

def create_env():
    env = gym.make(ENV)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return env, num_states, num_actions

def create_dqn_agent(num_states, num_actions):
    return DQNAgent(num_states, num_actions)

def run_cartpole():
    env, num_states, num_actions = create_env()
    dqn_agent = create_dqn_agent(num_states, num_actions)
    scores = deque(maxlen=NUM_SCORES)
    runtimes = []

    for epoch in range(EPOCHS):
        start_time = time.time()
        state = np.reshape(env.reset(), [1, num_states])

        for step in range(MAX_STEPS):
            action = dqn_agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            new_state = np.reshape(new_state, [1, num_states])

            dqn_agent.memoize(state, action, reward, new_state, done)
            state = new_state

            if done:
                scores.append(step)
                mean_score = round(np.mean(scores), 1)
                runtimes.append(start_time - time.time())
                print(f'Epoch: {epoch + 1}/1000 | Score: {step} | Avg Score: {mean_score} | Explore rate: {round(dqn_agent.epsilon, 4)}')

                if mean_score > WINNING_SCORE and len(scores) > NUM_SCORES:
                    print(f'Solved in {epoch} epochs with a mean score of {mean_score}')
                    print(f'Runtime: {np.sum(runtimes)} | Avg runtime: {np.mean(runtimes)}')
                    exit()

                if epoch == EPOCHS - 1:
                    print(f'Not solved. Mean score of last 100 epochs was {mean_score}')
                    print(f'Runtime: {np.sum(runtimes)} | Avg runtime: {np.mean(runtimes)}')

                break

            dqn_agent.replay()

run_cartpole()
