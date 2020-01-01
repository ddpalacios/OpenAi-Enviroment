from collections import namedtuple
import gym

# Lets start by opening our environment

# Tracking with Tuples
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

if __name__ == '__main__':
    steps = 0
    episode_reward = 0.0
    env = gym.make('CartPole-v0')
    observation = env.reset()  # Starting observation
    observation_size = env.observation_space.shape[0]  # Amount of observations being tracked
    action_size = env.action_space.n # Amount of potential actions. In our case --> [1,0]

    while True:
        # env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)  # Observation from action
        steps += 1
        episode_reward += reward

        reward_status = Episode(reward=reward, steps=steps)
        status = EpisodeStep(observation=obs, action=action)
        # print("{}{}\n".format(status, reward_status))
        if done:
            print("Earned {} rewards".format(episode_reward))
            env.close()
            break
