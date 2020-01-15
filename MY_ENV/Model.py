from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import adam
from Environment import Environment
import random
import sys
from Buffer import ExperienceBuffer

class model:
    def __init__(self, input_n, hidden_n, output_n):
        self.model = Sequential()
        self.model.add(Conv2D(input_n, kernel_size=(1, 1), activation='relu', input_shape=(1, 3,50,50)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(hidden_n, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(output_n, activation='softmax'))
        ADAM = adam(lr=0.01)
        self.model.compile(optimizer=ADAM, loss='mean_squared_error')

   