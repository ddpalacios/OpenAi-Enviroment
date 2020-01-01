import keras
from keras import optimizers, Sequential
from keras.layers import Dense
'''
Cross-entropy on CartPole using Keras

Our model is a one-hidden-layer neural network
'''


class Model:
    def __init__(self, hidden_size, batches, actions):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=batches))
        self.model.add(Dense(actions, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
