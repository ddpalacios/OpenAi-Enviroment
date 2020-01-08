#!/usr/bin/env python3
import gym
import collections
from time import sleep

ENV_NAME = "FrozenLake8x8-v0"
GAMMA = .9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)  # initialize our environment
        self.state = self.env.reset()  # Obtain our current state

        # Here are our tables we will be using in order
        # to retrieve past data (Dynamic Programming)
        #############################################
        self.rewards = collections.defaultdict(float)  # Rewards obtained
        self.transits = collections.defaultdict(collections.Counter)  # Use counter to keep track for probabilities
        self.values = collections.defaultdict(float)  # Our state values
        #############################################

    def play_n_random_steps(self, count):
        for _ in range(count):
            # We are collecting our standard data. Think about the graph
            # Diagram. We are simply "displaying" our data into our graph
            ######################################
            # But in order to do this, we need to apply some sample actions to our environment
            action = self.env.action_space.sample()  # Collect a random action
            new_state, reward, is_done, _ = self.env.step(action)  # Apply that random action

            # Saving Data to our tables
            ######################################################
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
            #######################################################

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            env.render()
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward

            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        ######################
        # Value iteration method
        ######################
        for state in range(self.env.observation_space.n): # For every possible state...
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)] # For every action...
            self.values[state] = max(state_values)
        '''
        It will be benifical to the agent if it knew the action values in advanced and be able to use it 
        to make our own decisions
        
        '''


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)  # To play test episodes

    # We initialize our environment, current state, and our tables for memoization
    agent = Agent()

    # Starting values
    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        # We will gather data from our env first. Think drawing our graph
        agent.play_n_random_steps(100)
        # Now that we gathered our data into our tables we will fill each state's action value
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
