from Agent import Blob
from Environment import Environment
import random
import sys
from time import sleep
if __name__ == '__main__':
    print(sys.argv)
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
       

    

    