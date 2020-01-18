from Environment import Environment
from Model import model
import sys
import numpy as np
import time
from time import sleep
AGGREGATE_STATS_EVERY = 50
EPSILON_FINAL = 0.02
total_rewards = []
EPSILON_START = 1.0
EPSILON_DECAY_LAST_FRAME = 10 ** 5
MIN_REWARD = -300  # For model save
MAX_REWARD = 300
GAMMA = .99
BATCH_SIZE = 32
MODEL_NAME = "DQN"
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1_000
ACTION_SPACE = 9
target_update_counter = 0
UPDATE_TARGET_EVERY = 5
###################
# Model Layer Param
# (FOR CONV2D)
###################
INPUT_N = 64
HIDDEN_N = 128
###################


def calc_loss(model, target_model, env, done):
    if env.exp_buffer._len_() < REPLAY_SIZE:
        return
    batch = env.exp_buffer.sample(BATCH_SIZE)
    current_states, actions, reward, dones, next_states = batch
    current_qs_list = model.predict(current_states)
    future_qs_list = target_model.predict(current_states)
    X_train = []
    y_train = []

    for idx, (current_state, action, reward, done, next_state) in enumerate(zip(current_states,
                                                                                actions,
                                                                                reward,
                                                                                dones,
                                                                                next_states)):
        if not done:
            max_future_q = np.max(future_qs_list[idx])
            new_q = reward + GAMMA * max_future_q
        else:
            new_q = reward

        current_qs = current_qs_list[idx]
        current_qs[action] = new_q
        X_train.append(current_state)
        y_train.append(current_qs)

    ##########################
    # TRAINING & WEIGHT TRANSFER
    ##########################
    target_model.fit(np.array(X_train)/255, np.array(y_train), batch_size= BATCH_SIZE, verbose=1,shuffle=False)
    if done:
        target_update_counter +=1
    if target_update_counter > UPDATE_TARGET_EVERY:
        print("SETTING WEIGHTS...")
        model.set_weights(target_model.get_weights())
        target_update_counter = 0
    ##########################


if __name__ == '__main__':
    episode = 0
    frame_idx = 0
    env = Environment(window_size=int(sys.argv[1]), step_size=int(sys.argv[2]), world_size=int(sys.argv[3]))
    show = int(sys.argv[4])
    state = env.reset()
    forward_prop = model(state.shape, INPUT_N, HIDDEN_N, ACTION_SPACE).model  # Forward propagation. Uses current weights and performs our linear algebra
    back_prop = model(state.shape, INPUT_N, HIDDEN_N, ACTION_SPACE).model  # For training (Calculates gradients with backpropagation)
    while True:
        frame_idx += 1
        if show:
            env.render()
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx/EPSILON_DECAY_LAST_FRAME)
        reward, is_done = env.play_step(forward_prop, epsilon, view_live_progress=False)
        calc_loss(forward_prop, back_prop, env, is_done)
        if is_done:
            episode+=1
            env.reset()

        if reward is not None:
            total_rewards.append(reward)
            average_reward = sum(total_rewards[-AGGREGATE_STATS_EVERY:])/len(total_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(total_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(total_rewards[-AGGREGATE_STATS_EVERY:])
            print("Min Reward: {}\nMax Reward: {}\nAverage Reward: {}\n Epsilon: {}".format(min_reward, max_reward, average_reward, epsilon))
            print("\n---------------------")
            if epsilon <= EPSILON_FINAL:
                print("...SAVING MODEL...")
                back_prop.model.save("DQN.h5")



