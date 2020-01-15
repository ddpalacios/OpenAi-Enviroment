from Environment import Environment
from Model import model
def calc_loss(batch, model, target_model):
    states, actions, reward, dones, next_states= batch
    next_state_values = target_model.predict(next_state_values)
    next_state_values = max(next_state_values)
    expected_state_action_values = next_state_values * GAMMA + reward
    model.fit(expected_state_action_values)
    


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
