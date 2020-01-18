from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import Adam
import numpy as np
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
        self.model.add(Dense(512, activation= 'relu'))
        self.model.add(Dense(output_n, activation='linear'))
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    def t_pred(self, state):
        predicted = self.model.predict(state)
        return predicted

    def Predict(self, state):
        predicted = self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
        return np.array(predicted)

    def Train(self, X_train, y_train, BATCH_SIZE, model,tgt_model, done):
      
        self.model.fit(X_train/255, y_train, batch_size= BATCH_SIZE, verbose=1,shuffle=False)
        if done:
            self.target_update_counter +=1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            print("SETTING WEIGHTS...")
            model.model.set_weights(tgt_model.get_weights())
            self.target_update_counter = 0



