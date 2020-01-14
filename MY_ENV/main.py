from Agent import Blob
from Environment import Environment
import random
from time import sleep
if __name__ == '__main__':
    test_env = Environment()
    state = test_env.reset()
    while True:
        action = test_env.action_sample()
        test_env.render()
        state, reward, is_done = test_env.step(action)
        print(action)
    

    

    