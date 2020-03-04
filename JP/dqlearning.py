from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np

import pandas as pd

class DeepQLearner() :

    def __init__(self) :
        super().__init__()
        self._learning = True
        self._learning_rate = 0.1
        self._discount = 0.1
        self._epsilon = 0.9

        # Create model
        model = Sequential()
        model.add(Dense(2, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(10, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(4, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='adam')

        self._model = model
    
    # Choose an action based on the model
    def getAction(self, state) :
        rewards = self._model.predict(np.array([state]), batch_size=1)

        rd = np.random.uniform(0,1)
        if rd < self._epsilon :
            if rewards[0][0] > rewards[0][1] :
                action = 1
            else :
                action = 0
        
        else :
            action = np.random.randint(2)
        
        self._last_state = state
        self._last_action = action
        self._last_target = rewards

        return action
    
    # Update the table thanks to the intelligent model
    def update(self, new_state, reward) :
        if self._learning :
            rewards = self._model.predict(np.array([new_state]), batch_size=1)
            if rewards[0][0] > rewards[0][1] :
                maxQ = rewards[0][0]
            else :
                maxQ = rewards[0][1]
            
            new = self._discount * maxQ

            if self._last_action == 1 :
                self._last_target[0][0] = reward + new
            else :
                self._last_target[0][1] = reward + new
            
            self._model.fit(np.array([self._last_state]), self._last_target, batch_size = 1, epochs = 1, verbose = 0)
