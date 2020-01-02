import keras
from keras import optimizers, Sequential
from keras.layers import Dense
import numpy as np

'''
Cross-entropy on CartPole using Keras

Our model is a one-hidden-layer neural network
'''


class Model:
    def __init__(self, observation_size, hidden_size, actions_n):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=observation_size))
        self.model.add(Dense(actions_n, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['acc'])

    def select_action(self, observations):  # Just predicts an action
        action_probabilities = self.model.predict(np.asarray([observations]), verbose=0)
        return np.random.choice(len(action_probabilities[0]), p=action_probabilities[0])  # Random Sampling

    def train(self, train_observations, train_actions):
        self.model.fit(train_observations, train_actions, epochs=50, verbose=0)
