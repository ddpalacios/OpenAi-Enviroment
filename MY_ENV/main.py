from Environment import Environment
from Model import model
import sys
import numpy as np
import time
from time import sleep
from tqdm import tqdm

EPISODE_MAX = 20000
AGGREGATE_STATS_EVERY = 50
EPSILON_FINAL = 0.01
total_rewards = []
EPSILON_START = 1.0
EPSILON_DECAY_LAST_FRAME = 10 ** 5
MIN_REWARD = -200  # For model save
MAX_REWARD = 300
GAMMA = .99
BATCH_SIZE = 64
MODEL_NAME = "DQN"
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
target_update_counter = 0
UPDATE_TARGET_EVERY = 5
###################
# Model Layer Param
# (FOR CONV2D)
###################
INPUT_N = 256
HIDDEN_N = 256
ACTION_SPACE = 9
##################

def calc_loss(main_model, target_model, env, is_done):
    if env.exp_buffer._len_() < REPLAY_SIZE:
        return
    X_train = []
    y_train = []
    batch = env.exp_buffer.sample(BATCH_SIZE)
    current_states, actions, reward, dones, next_states = batch
    current_qs_list = main_model.model.predict(current_states / 255)
    future_qs_list = target_model.model.predict(next_states / 255)

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

    main_model.model.fit(np.array(X_train) / 255, np.array(y_train), batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    if is_done:
        main_model.target_update_counter += 1
    if main_model.target_update_counter > UPDATE_TARGET_EVERY:
        print("SETTING WEIGHTS...")
        target_model.model.set_weights(main_model.model.get_weights())
        main_model.target_update_counter = 0


if __name__ == '__main__':
    episode = 0
    epsilon = 1.0
    MIN = 0.0
    frame_idx = 0
    env = Environment(window_size=int(sys.argv[1]), step_size=int(sys.argv[2]), world_size=int(sys.argv[3]))
    show = int(sys.argv[4])
    state = env.reset()
    main_model = model(state.shape, INPUT_N, HIDDEN_N,
                       ACTION_SPACE)  # Forward propagation. Uses current weights and performs our linear algebra
    target_model = model(state.shape, INPUT_N, HIDDEN_N,
                         ACTION_SPACE)  # For training (Calculates gradients with backpropagation)
    target_model.model.set_weights(main_model.model.get_weights())

    while True:
        frame_idx += 1
        if show:
            env.render()

        reward, is_done = env.play_step(target_model, epsilon, view_live_progress=False)
        calc_loss(main_model, target_model, env, is_done)
        if is_done:
            episode += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            episode += 1
            env.reset()

        if reward is not None:
            total_rewards.append(reward)
            average_reward = sum(total_rewards[-AGGREGATE_STATS_EVERY:]) / len(total_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(total_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(total_rewards[-AGGREGATE_STATS_EVERY:])
            print("Total reward from latest episode #{}: {}\n"
                  "Min Reward (Last 50): {}\n"
                  "Max Reward (Last 50): {}\n"
                  "Average Reward: {}\n"
                  "Epsilon: {}".format(episode, reward, min_reward,
                                       max_reward,
                                       average_reward,
                                       epsilon))
            print("\n---------------------")
            if min_reward >= MIN_REWARD:
                print("...SAVING MODEL...")
                target_model.model.save("DQN Min:{} | Max: {} | Average: {}.h5".format(min_reward,
                                                                                       max_reward,
                                                                                       average_reward))

            if episode >= EPISODE_MAX:
                print("...SAVING MODEL...")
                target_model.model.save("DQN Min:{} | Max: {} | Average: {}.h5".format(min_reward,
                                                                                       max_reward,
                                                                                       average_reward))
                break
