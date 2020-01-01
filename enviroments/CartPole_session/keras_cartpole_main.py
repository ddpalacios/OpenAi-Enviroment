from collections import namedtuple

import gym

# Lets start by opening our environment

# Tracking with Tuples
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, model, batch_size): # Batches are the amount of episodes given
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observation = env.reset()

    finished = False
    while not finished: # While the episode is not finished...
        action = model.select_action(observation) # Select an action based on its current observation
        next_observation, reward, done, _ = env.step(action) # Apply action and apply next observation
        episode_reward += reward # Sum up each reward (How long can the pole be balanced?)
        episode_steps.append(EpisodeStep(observation=observation, action=action)) # Using collections tuple to track each observation and action taken
        finished = done # Boolean --> True --> breaks out of loop
        observation = next_observation
    return Episode(reward=episode_reward, steps=episode_steps)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()  # Starting observation
    observation_size = env.observation_space.shape[0]  # Amount of observations being tracked
    action_size = env.action_space.n  # Amount of potential actions. In our case --> [1,0]

    # while True:
    #     action = env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)  # Observation from action
    #     steps += 1
    #     episode_reward += reward
    #
    #     reward_status = Episode(reward=reward, steps=steps)
    #     status = EpisodeStep(observation=obs, action=action)
    #     # print("{}{}\n".format(status, reward_status))
    #     if done:
    #         print("Earned {} rewards".format(episode_reward))
    #         env.close()
    #         break
