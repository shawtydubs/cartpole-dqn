import gym
import numpy as np

from dqn import DQNAgent

np.random.seed(1)

def create_env():
    env = gym.make('CartPole-v0')
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return env, num_states, num_actions

def create_dqn_agent(num_states, num_actions):
    return DQNAgent(num_states, num_actions)

def run_cartpole():
    env, num_states, num_actions = create_env()
    dqn_agent = create_dqn_agent(num_states, num_actions)

    for epoch in range(1000):
        state = np.reshape(env.reset(), [1, num_states])

        for step in range(200):
            action = dqn_agent.get_action(state)
            new_state, reward, done, _ = env.step(action)

            reward = reward if not done else -reward
            new_state = np.reshape(new_state, [1, num_states])

            dqn_agent.memoize(state, action, reward, new_state, done)
            state = new_state

            if done:
                print(f'Epoch: {epoch}/1000 | Step: {step}')
                break

            dqn_agent.replay()

run_cartpole()
