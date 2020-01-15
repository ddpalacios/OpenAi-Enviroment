from Environment import Environment
from Model import model
import sys
from time import sleep

EPSILON_FINAL = 0.02
EPSILON_START = 1.0
EPSILONE_DECAY_LAST_FRAME = 10 ** 5
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
    frame_idx = 0
    env = Environment(window_size=int(sys.argv[1]), step_size=int(sys.argv[2]), world_size=int(sys.argv[3]))
    show = int(sys.argv[4])
    state = env.reset()
    model = model(state.shape, INPUT_N, HIDDEN_N, 9)
    while True:
        action = model.Predict(state)

        if show:
            env.render()

        state, reward, is_done = env.step(action)
        if is_done:
            break

        # frame_idx +=1
    #     epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    #     reward = env.play_step(model, epsilon)
    #     if reward is not None:
    #         total_rewards.append(reward)
    #         calc_loss(batch, model, tgt_model)
