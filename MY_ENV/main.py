from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import adam
from Environment import Environment
import random
import sys

class model:
    def __init__(self, input, input_n, hidden_n, output_n):
        self.model = Sequential()
        self.model.add(Conv2D(input_n, kernel_size=(3, 3), activation='relu', input_shape=input.shape))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(hidden_n, activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(output_n, activation='softmax'))
        ADAM = adam(lr=0.01)
        self.model.compile(optimizer=ADAM, loss='mean_squared_error')

    def predict(self):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    test_env = Environment(window_size=int(sys.argv[1]), step_size=int(sys.argv[2]), world_size=int(sys.argv[3]))
    show = int(sys.argv[4])
    state = test_env.reset()
    while True:
        action = test_env.action_sample()
        if show:
            test_env.render()
        state, reward, is_done = test_env.step(action)
        if is_done:
            break
