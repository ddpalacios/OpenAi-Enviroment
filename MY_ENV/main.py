from Environment import Environment
from Model import model
import sys
import numpy as np
from time import sleep

EPSILON_FINAL = 0.02
total_rewards = []
EPSILON_START = 1.0
EPSILON_DECAY_LAST_FRAME = 10 ** 5
GAMMA = .99
BATCH_SIZE = 32
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1_000

###################
# Model Layer Param
# (FOR CONV2D)
###################
INPUT_N = 512
HIDDEN_N = 128
###################


def calc_loss(batch, forward_prop, back_prop, env):
    X_train = []
    y_train = []
    current_states, actions, reward, dones, next_states = batch
    for idx, (current_state, action, reward, done, next_state) in enumerate(zip(current_states,
                                                                                actions,
                                                                                reward,
                                                                                dones,
                                                                                next_states)):
        state_action_values = forward_prop.Predict(current_state)
        future_q_list = forward_prop.Predict(next_state)
        #
        # if not done:
        #     max_future_q = np.max(future_q_list)
        #     expected_state_action_values = reward + GAMMA * max_future_q
        # else:
        #     expected_state_action_values = reward  # Immediate reward
        #
        # # Update Q value for a given state
        # current_qs = current_states[idx]
        #
        # X_train.append(current_state)
        # y_train.append(current_qs)

    # back_prop.Train(expected_state_action_values, env)


if __name__ == '__main__':
    episode = 0
    frame_idx = 0
    env = Environment(window_size=int(sys.argv[1]), step_size=int(sys.argv[2]), world_size=int(sys.argv[3]))
    show = int(sys.argv[4])
    state = env.reset()
    forward_prop = model(state.shape, INPUT_N, HIDDEN_N,
                         9)  # Forward propagation. Uses current weights and performs our linear algebra
    back_prop = model(state.shape, INPUT_N, HIDDEN_N, 9)  # For training (Calculates gradients with backpropagation)
    while True:
        frame_idx += 1
        if show:
            env.render()
        epsilon = 0
        reward, is_done = env.play_step(forward_prop, epsilon, view_live_progress=False)

        if reward is not None:
            total_rewards.append(reward)
            batch = env.exp_buffer.sample(BATCH_SIZE)

            if is_done:
                episode += 1
                print("\n... EPISODE IS DONE")
                env.reset()

            calc_loss(batch, forward_prop, back_prop, env)
