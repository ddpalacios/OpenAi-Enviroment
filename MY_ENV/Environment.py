import turtle
import numpy as np
from PIL import Image
import cv2
from Agent import Blob
import random

class Environment:
    def __init__(self, title="Blob world"):
        self.actions = [0,1,2,3,4,5,6,7,8]
        self.title = title
        self.SIZE = 10
        self.RETURN_IMAGES = True
        self.MOVE_PENALTY = 1
        self.ENEMY_PENALTY = 300
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
        img = img.resize((300,300))
        cv2.imshow("Image", np.array(img))
        cv2.waitKey(1)
        
    def step(self, action):
        self.episode_step +=1
        self.player.action(action)
        if self.RETURN_IMAGES:
            new_observation=np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY
        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step > 20:
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
