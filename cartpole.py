import gym
import numpy as np
from collections import deque

from dqn import DQNAgent

EPOCHS = 1000
MAX_STEPS = 500
NUM_SCORES = 100
WINNING_SCORE = 475

def create_env():
    env = gym.make('CartPole-v1')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return env, num_states, num_actions

def create_dqn_agent(num_states, num_actions):
    return DQNAgent(num_states, num_actions)

def run_cartpole():
    env, num_states, num_actions = create_env()
    dqn_agent = create_dqn_agent(num_states, num_actions)
    scores = deque(maxlen=NUM_SCORES)

    for epoch in range(EPOCHS):
        state = np.reshape(env.reset(), [1, num_states])

        for step in range(MAX_STEPS):
            action = dqn_agent.get_action(state)
            new_state, reward, done, _ = env.step(action)

            reward = reward if not done else -reward
            new_state = np.reshape(new_state, [1, num_states])

            dqn_agent.memoize(state, action, reward, new_state, done)
            state = new_state

            if done:
                scores.append(step)
                mean_score = round(np.mean(scores), 1)
                print(f'Epoch: {epoch}/1000 | Last step: {step} | Avg Step: {mean_score} | Explore rate: {dqn_agent.epsilon}')

                if mean_score > WINNING_SCORE and len(scores) > NUM_SCORES:
                    print(f'Solved in {epoch} epochs with a mean score of {mean_score}')
                    exit()

                break

            dqn_agent.replay()

run_cartpole()
