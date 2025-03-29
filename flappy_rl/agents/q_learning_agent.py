import numpy as np
import matplotlib.pyplot as plt
from ple import PLE
from ple.games.flappybird import FlappyBird

n_states = 15 * 15 * 15 * 5 # Total number of discrete states (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)
n_actions = 2 # The actions are either 0 (flap the wings) or 1 (do nothing)
n_bins = 15 # Used for position binning

learning_rate = 0.1
epsilon = 0.1
discount_factor = 0.9
epochs = 1000 # Number of training episodes

Q_table = np.zeros((n_states, n_actions))

game = FlappyBird()
env = PLE(game, fps=30, display_screen=True)
env.init()

def clip(val):
    # Clip the value to the range [0, 40]
    return min(max(int(val / 40), 0), 14)

def get_discrete_state(state):
    player_y = clip(state['player_y']) # The y position of the bird
    next_pipe_top_y = clip(state['next_pipe_top_y']) # The top y position of the next gap
    next_pipe_dist_to_player = clip(state['next_pipe_dist_to_player']) # The horizontal distance between bird and next pipe
    player_vel = min(max(state['player_vel'] + 8, 0), 18) # The current vertical velocity of the bird - shift from [-8,10] to [0,18]
    vel_bin = min(int(player_vel / 4), 4) # The velocity is divided into 5 bins (0-4)
    return (player_y * 15 * 15 * 5 +
            next_pipe_top_y * 15 * 5 + 
            next_pipe_dist_to_player * 5 + 
            vel_bin) 

rewards = []
for epoch in range(epochs):
    env.reset_game()
    total_reward = 0

    while not env.game_over():
        state = get_discrete_state(env.getGameState())

        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1]) # Explore: choose a random action
        else:
            action = np.argmax(Q_table[state]) # Exploit: choose the action with the highest Q-value

        reward = env.act(action)
        if env.game_over():
            reward = -5
        else:
            reward = 1

        total_reward += reward
        next_state = get_discrete_state(env.getGameState())

        Q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action]
        )

    print(f"Episode {epoch + 1}: Total reward = {total_reward}")
    rewards.append(total_reward)

print(rewards)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
 