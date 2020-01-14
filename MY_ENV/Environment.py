import turtle
import numpy as np
from PIL import Image
import cv2
from Agent import Blob
import random
from time import sleep
class Environment:
    def __init__(self,window_size,step_size, world_size=10,title="Blob world"):
        self.actions = [0,1,2,3,4,5,6,7,8]
        self.window_size = window_size
        self.MAX_AMOUNT_OF_STEPS = step_size
        self.title = title
        self.SIZE = world_size
        self.RETURN_IMAGES = True
        self.MOVE_PENALTY = -1
        self.ENEMY_PENALTY = -300
        self.FOOD_REWARD = 25
        self.OBSERVATION_SPACE_VALUES = (self.SIZE, self.SIZE, 3)  # 4
        self.ACTION_SPACE_SIZE = 9
        self.PLAYER_N = 1  # player key in dict
        self.FOOD_N = 2  # food key in dict
        self.ENEMY_N = 3  # enemy key in dict
        # the dict! (colors)
        self.d = {1: (255, 175, 0),
            2: (0, 255, 0),
            3: (0, 0, 255)}

    def action_sample(self):
        action = random.choice(self.actions)
        return action
           
    def render(self):
        img = self.get_image()
        img = img.resize((self.window_size,self.window_size))
        cv2.imshow(self.title, np.array(img))
        cv2.waitKey(1)
        
    def step(self, action):
        reward = 0.0
        done = False
        self.episode_step +=1
        self.player.action(action)
        new_observation=np.array(self.get_image())
      
        if self.player == self.enemy:
            reward += self.ENEMY_PENALTY
            print("HIT ENEMY\nReward:",reward)
            sleep(2)
        elif self.player == self.food:
            reward += self.FOOD_REWARD
            print("OBTAINED FOOD\nReward:",reward)
            sleep(2)
            self.food.x = np.random.randint(0, self.SIZE)
            self.food.y = np.random.randint(0, self.SIZE)
        else:
            reward += self.MOVE_PENALTY
       

        if reward == self.ENEMY_PENALTY or self.episode_step > self.MAX_AMOUNT_OF_STEPS:
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
        self.episode_step = 0
        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        return observation
