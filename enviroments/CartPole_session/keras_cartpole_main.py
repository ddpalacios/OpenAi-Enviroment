from collections import namedtuple
from BuildingRLModelKeras import Model
import gym
import numpy as np
# Lets start by opening our environment

# Tracking with Tuples
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, model, render=False): # Batches are the amount of episodes given
    batch = []
    episode_reward = 0.0
    episode_steps = []
    observation = env.reset()


    finished = False
    while not finished: # While the episode is not finished...
        if render:
            env.render() # To display
        action = model.select_action(observation) # Select an action based on its current observation
        next_observation, reward, done, _ = env.step(action) # Apply action and apply next observation
        episode_reward += reward # Sum up each reward (How long can the pole be balanced?)
        episode_steps.append(EpisodeStep(observation=observation, action=action)) # Using collections tuple to track each observation and action taken
        finished = done # Boolean --> True --> breaks out of loop
        observation = next_observation
    return Episode(reward=episode_reward, steps=episode_steps)
def iterate(env, model, batch_size, render=False):
    batch = []
    while True:
        episode_result = iterate_batches(env, model, render)
        batch.append(episode_result)
        if len(batch) == batch_size:
            yield batch
            batch = []
def fulfills(batch, percentile):
    rewards = list(map(lambda step: step.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound:
            continue

        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    return train_obs, train_act, reward_bound, reward_mean



###############################
# HERE ARE YOUR HYPERPARAMETERS
# Use these variables to test out
# your RL model!
###############################

##############
# TUNE ME
###############
PERCENTILE = 70
batch_size = 16
SHOW = True
hidden_size = 128
################


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()  # Starting observation
    observation_size = env.observation_space.shape[0]  # Amount of observations being tracked
    action_size = env.action_space.n  # Amount of potential actions. In our case --> [1,0]
    model = Model(observation_size , hidden_size ,action_size)

    for idx, batch in enumerate(iterate(env, model,batch_size, SHOW)):
        obs_v, acts_v, reward_b, reward_m = fulfills(batch, PERCENTILE)
        model.train(np.asarray(obs_v), np.asarray(acts_v))
        if reward_m > 199:
            print("Solved!")
            break

        print(f"{idx}:  reward_mean={reward_m}, reward_bound={reward_b}")

    iterate_batches(env, model, render=True)
    env.close()
