from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import Adam

UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)


class model:
    def __init__(self, input_shape, input_n, hidden_n, output_n):
        self.target_update_counter = 0
        self.model = Sequential()
        self.model.add(Conv2D(input_n, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(.2))
        self.model.add(Conv2D(hidden_n, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(.2))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Dense(output_n, activation='linear'))
        self.model.compile(optimizer=Adam(lr=.001), loss='mse', metrics=['accuracy'])
