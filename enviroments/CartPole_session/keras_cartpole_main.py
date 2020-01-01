from collections import namedtuple

import gym

# Lets start by opening our environment

# Tracking with Tuples
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

if __name__ == '__main__':
    steps = 0
    env = gym.make('CartPole-v0')
    observation = env.reset()  # Starting observation

    while True:
        env.render()
        action = env.action_space.sample()
        # if action:
        #     print("Going right")
        # else:
        #     print("Going left")

        obs, reward, done, _ = env.step(action)  # Observation from action
        steps +=1
        reward_status = Episode(reward=reward, steps=steps)
        status = EpisodeStep(observation=obs, action=action)
        print("{}{}\n".format(status, reward_status))
        #
        # if done:
        #     env.close()
        #
        #     break
