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
    observation = env.reset() # Getting current observation


    finished = False
    while not finished: # While the episode is not finished...

        if render:
            env.render() # To display

        '''
        This chunk of code calls a method from our class of our model called 'select_action'.
        Looking back, this method simply runs through our model with our current weights and gives us our predection actions [1/0]
        
        This predicted action is then ran through gym .step() method which does the action and returns
        its observation based on its action, reward, and boolean 'done' to indicate that the pole has fallen and the next episode begins
        
        Once we sum up our rewards, we use tuples to track its current observation and action and save it with a list
        
        We then break out of this loop and return the tracking record for its Episode. 
        Notice that once episode contains the total reward and the steps it took to achieve this award.  
        and these steps are what contain 1). the action it took   2). Based on its action, what is its observation
        '''
        ############################################################################################
        action = model.select_action(observation) # Select an action based on its current observation
        next_observation, reward, done, _ = env.step(action) # Apply action and apply next observation
        episode_reward += reward # Sum up each reward (How long can the pole be balanced?) for current episode
        episode_steps.append(EpisodeStep(observation=observation, action=action)) # Using collections tuple to track each observation and action taken
        finished = done # Boolean --> True --> breaks out of loop
        observation = next_observation
        #############################################################################################

    # Now lets head back to our original method 'iterate' with this in mind...
    return Episode(reward=episode_reward, steps=episode_steps)

def iterate(env, model, batch_size, render=False):
    batch = []
    while True:
        episode_result = iterate_batches(env, model, render) # We have our episode result
        batch.append(episode_result) # that is consider 1 out n batch size
        if len(batch) == batch_size: # Once we have reached our batch size limit, we want to yield our batch
            yield batch # This is for iteration purpuses. Google 'return' VS 'yield'
            batch = [] # Once we have iterated through our entire batch, lets empty it and get ready for our next batch


def fulfills(batch, percentile):
    rewards = list(map(lambda step: step.reward, batch)) # Mapping our batch and from each episode, we obtain its reward and saving it into a list named rewards
    reward_bound = np.percentile(rewards, percentile) # Returns percentile based on list of reward values
    reward_mean = float(np.mean(rewards)) # This is for us. just to view its progress by obtaining its mean

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound: # Discard any rewards that are not >= reward bound
            continue

        train_obs.extend(map(lambda step: step.observation, example.steps)) # Save the episodes obsercvations
        train_act.extend(map(lambda step: step.action, example.steps)) # Save the episodes actions

    # Now that we have discarded any low rewards and saved the best fit ones, we will now return such values
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
SHOW = False
hidden_size = 64
################


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0') # Selecting our enviroment
    observation = env.reset()  # Starting observation

    # Initilizing model, observation and action size
    #########################################################
    observation_size = env.observation_space.shape[0]  # Amount of observations being tracked
    action_size = env.action_space.n  # Amount of potential actions. In our case --> [1,0]
    model = Model(observation_size , hidden_size ,action_size)
    ##########################################################

                            # Training Loop
    #####################################################################
    for idx, batch in enumerate(iterate(env, model,batch_size, SHOW)): # This method collects our batches (Episode results)
        obs_v, acts_v, reward_b, reward_m = fulfills(batch, PERCENTILE) # Now think back to our supervised learning techniques, we have our data and our target values
                                                                        # This is now starting to feel like your classic classification problem
        model.train(np.asarray(obs_v), np.asarray(acts_v)) # We will convert these values as an numpy array and start training our network!
        if reward_m > 199: # If our mean has reached its max achievment value, we have now solved this problem
            print("Solved!")
            break
        print(f"{idx}:  reward_mean={reward_m}, reward_bound={reward_b}")
    #####################################################################


    # Once all done, we can see its ending result
    #######################################
    iterate_batches(env, model, render=True)
    env.close()
    #######################################

