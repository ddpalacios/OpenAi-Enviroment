from Environment import Environment
from Model import model
import sys
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
# Model Layer Param (FOR CONV2D)
###################
INPUT_N = 512
HIDDEN_N = 128
###################


def calc_loss(batch, model, target_model):
    states, actions, reward, dones, next_states = batch
    next_state_values = target_model.predict(states)
    next_state_values = max(next_state_values)
    expected_state_action_values = next_state_values * GAMMA + reward
    model.fit(expected_state_action_values)


if __name__ == '__main__':
    episode =0
    frame_idx = 0
    env = Environment(window_size=int(sys.argv[1]), step_size=int(sys.argv[2]), world_size=int(sys.argv[3]))
    show = int(sys.argv[4])
    state = env.reset()
    forward_prop = model(state.shape, INPUT_N, HIDDEN_N, 9) # Forward propagation. Uses current weights and performs our linear algebra
    back_prop = model(state.shape, INPUT_N, HIDDEN_N, 9) # For training (Calculates gradients with backpropagation)
    while True:
        frame_idx += 1
        if show:
            env.render()
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        print("... EPSILON DECAY", epsilon)
        reward, is_done = env.play_step(forward_prop, epsilon)

        if reward is not None:
            total_rewards.append(reward)
            if is_done:
                episode+=1
                print("\n... EPSILON DECAY", epsilon)
                # sleep(.5)
                print("Rewards from episode #{}:\n{}\n".format(episode, total_rewards))
                env.reset()
            # calc_loss(batch, model, tgt_model)
