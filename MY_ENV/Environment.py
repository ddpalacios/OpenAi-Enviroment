import numpy as np
from PIL import Image
import cv2
from Agent import Blob
import random
import collections
from Buffer import ExperienceBuffer

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Environment:
    def __init__(self, window_size, step_size, world_size=10, REPLAY_SIZE=10_000, title="Blob world"):
        self.episode_reward = 0.0  # Lets first initialize our reward to 0
        self.episode_step = 0
        self.total_reward = 0.0
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.window_size = window_size
        self.MAX_AMOUNT_OF_STEPS = step_size
        self.title = title
        self.REPLAY_SIZE = REPLAY_SIZE
        self.exp_buffer = ExperienceBuffer(REPLAY_SIZE)
        self.SIZE = world_size
        self.RETURN_IMAGES = True
        self.MOVE_PENALTY = -1
        self.ENEMY_PENALTY = -300
        self.NO_PROGRESS_PENALITY = -100
        self.FOOD_REWARD = 300
        self.OBSERVATION_SPACE_VALUES = (self.SIZE, self.SIZE, 3)  # 4
        self.ACTION_SPACE_SIZE = 9
        self.PLAYER_N = 1  # player key in dict
        self.FOOD_N = 2  # food key in dict
        self.ENEMY_N = 3  # enemy key in dict
        # the dict! (colors)
        self.d = {1: (255, 175, 0),
                  2: (0, 255, 0),
                  3: (0, 0, 255)}
        self.state = self.reset()

    def _reset(self):
        self.total_reward = 0.0

    def play_step(self, model, epsilon=0.0, view_live_progress=False):
        done_reward = None
        if np.random.random() < epsilon:  # Epsilon Decay --> Exploration vs Exploitation Dilemma
            action = self.action_sample()  # Choosing a random action n% of the time
        else:
            state = self.state  # Otherwise, get its current state
            # print("... Forward prop action calculating...")
            q_val = model.Predict(state)  # And using models current weights, perform your forward prop
            action = np.argmax(q_val)  # Then retrieve the index with its maximum value

        new_state, reward, is_done = self.step(
            action)  # Perform chosen action and return its new state, its reward, and our boolean
        self.total_reward += reward  # Add the given reward to our total reward
        new_state = new_state

        if view_live_progress:
            self.track_progress(action, reward, is_done, new_state, epsilon)

        exp = Experience(self.state, action, reward, is_done, new_state)  # Obtain our (s,a,r, done ,s')
        self.exp_buffer.append(exp)  # Append this tuple into our replay buffer. this will be used for our training data
        self.state = new_state  # Then we will update the new state with its current state
        if is_done:  # If our episode is done
            done_reward = self.total_reward  # Obtain the finished rewards that was obtained during the episode
            self._reset()  # Reset our environment. Reset total rewards and state

        return done_reward, is_done  # Finally, we will return our reward back to our main file

    def action_sample(self):
        action = random.choice(self.actions)
        return action

    def track_progress(self, action, reward, is_done, new_state, epsilon):
        # Live Progress
        #######################################
        print(
            "\n\nEXPERIENCE BUFFER:\n"
            "STEP# {}:\n"
            "EPSILON DECAY: {}\n"
            "Original state (shape): {}\n"
            "Action: {}\nReward given: {}\n"
            "is done?: {}\nNew State: {}\n" \
                .format(self.episode_step, epsilon,
                        self.state.shape, action, reward,
                        is_done, new_state.shape))
        print("REPLAY BUFFER LENGTH: {}".format(self.exp_buffer._len_()))
        if self.exp_buffer._len_() >= 10_000:
            print("\nNo longer appending expierience...\n...Ready to train data with current buffer...")
        print("\n\n----------------")
        #########################################

    def render(self):
        img = self.get_image()
        img = img.resize((self.window_size, self.window_size))
        cv2.imshow(self.title, np.array(img))
        cv2.waitKey(1)

    def step(self, action):
        '''
        :param action --> Indicated action [1 ... 8]:
        :return: new_state, reward, is_done

        This function will allow us to interact with our enviroment with
        its indicated chosen action
        '''
        done = False  # Our flag to indicate when an episode is over
        self.episode_step += 1  # Amount of steps taken in our episode
        # print("Steps taken:", self.episode_step)

        self.player.action(action)  # Perform that step to our agents class (Dont worry about how this works. \
        # Just know it will perform that specified action updating (x,y) cord)

        new_observation = np.array(
            self.get_image())  # From the current coordinates, retrieve the RGB image for our Conv2D model
        if self.player == self.enemy:  # if the player hits the enemy
            reward = self.ENEMY_PENALTY  # add the penality to our current reward
            print("HIT ENEMY\nReward:", reward)
        elif self.player == self.food:
            reward = self.FOOD_REWARD
            print("OBTAINED FOOD\nReward:", reward)
            # self.food.x = np.random.randint(0, self.SIZE)
            # self.food.y = np.random.randint(0, self.SIZE)
        elif self.episode_step >= self.MAX_AMOUNT_OF_STEPS:
            reward = self.NO_PROGRESS_PENALITY
        else:
            reward = self.MOVE_PENALTY

        if reward == self.FOOD_REWARD or reward == self.ENEMY_PENALTY or self.episode_step > self.MAX_AMOUNT_OF_STEPS:
            done = True

        return new_observation, reward, done

    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')
        return img

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)
        observation = np.array(self.get_image())
        self.episode_step = 0

        return observation
