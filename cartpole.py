import gym
import numpy as np
import time
from collections import deque

from dqn import DQNAgent

ENV = 'CartPole-v0'
WINNING_SCORE = 195
MAX_STEPS = 200

# ENV = 'CartPole-v1'
# WINNING_SCORE = 475
# MAX_STEPS = 500

EPOCHS = 10000
NUM_SCORES = 100

def create_env():
    env = gym.make(ENV)
    env.seed(1)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return env, num_states, num_actions

def create_dqn_agent(num_states, num_actions):
    return DQNAgent(num_states, num_actions)

def end_step(epoch, step, start_time, max_avg_score, scores, runtimes):
    epoch_score = step + 1
    scores.append(epoch_score)
    mean_score = round(np.mean(scores), 1)
    max_avg_score = max(max_avg_score, mean_score)
    runtimes.append(time.time() - start_time)
    print(f'Epoch: {epoch + 1}/{EPOCHS} | Epoch Score: {epoch_score} | Avg Score: {mean_score} | Max Avg Score: {max_avg_score}')

    if mean_score >= WINNING_SCORE and len(scores) >= NUM_SCORES:
        print(f'Solved in {epoch} epochs with a mean score of {mean_score}')
        print(f'Runtime: {np.sum(runtimes)} | Avg runtime: {np.mean(runtimes)}')
        exit()

    if epoch == EPOCHS - 1:
        print(f'Not solved.')
        print(f'Runtime: {np.sum(runtimes)} | Avg runtime: {np.mean(runtimes)}')

    return scores, runtimes, max_avg_score

def run_cartpole():
    env, num_states, num_actions = create_env()
    dqn_agent = create_dqn_agent(num_states, num_actions)
    scores = deque(maxlen=NUM_SCORES)
    runtimes = []
    max_avg_score = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        state = np.reshape(env.reset(), [1, num_states])

        for step in range(MAX_STEPS):
            action = dqn_agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            new_state = np.reshape(new_state, [1, num_states])

            dqn_agent.memoize(state, action, reward, new_state, done)
            state = new_state

            if done:
                scores, runtimes, max_avg_score = end_step(epoch, step, start_time, max_avg_score, scores, runtimes)
                break

            dqn_agent.replay()

run_cartpole()
