from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import Adam
import numpy as np


class model:
    def __init__(self, input_shape, input_n, hidden_n, output_n):
        self.model = Sequential()
        self.model.add(Conv2D(input_n, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(hidden_n, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(output_n, activation='softmax'))
        self.model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

    def Predict(self, state):
        predicted = self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
        predicted = np.argmax(predicted)  # Returns the INDEX of the maximum value passed in array
        return predicted
