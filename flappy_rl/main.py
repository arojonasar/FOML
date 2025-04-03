import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from collections import deque
from ple import PLE
from ple.games.flappybird import FlappyBird

# Hyperparameters
learning_rate = 0.0001 #0.01 -> 0.0001
initial_epsilon = 1.0  
min_epsilon = 0.05    
epsilon_decay = 0.999995 #0.99 -> 0.999
discount_factor = 0.99 #0.9 -> 0.99
epochs = 5000
max_replay_size = 50000 #1000 -> 50000
batch_size = 32
target_update_freq = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)
env.init()

def normalize_feature(x, min_val, max_val):
    return 2 * (x - min_val) / (max_val - min_val) - 1

def normalize_state(state):
    return np.array([
        normalize_feature(state['player_y'], 0, 512),
        normalize_feature(state['next_pipe_top_y'], 0, 512),
        normalize_feature(state['next_pipe_dist_to_player'], 0, 288),
        normalize_feature(state['player_vel'], -8, 10)
    ], dtype=np.float32)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 2)
        #sigmoid -> LEAKY
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky(self.fc1(x))
        x = self.leaky(self.fc2(x))
        return self.fc3(x)

main_model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(main_model.state_dict())
target_model.eval()

optimizer = optim.Adam(main_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
replay_buffer = deque(maxlen=max_replay_size)
actions = env.getActionSet()

episode_rewards = []
epsilon = initial_epsilon
step_count = 0

for file in glob.glob("flappy_dqn_episode_*.pth"):
    os.remove(file)
print("Old model checkpoints deleted.")

for episode in range(epochs):
    env.reset_game()
    total_reward = 0
    state = normalize_state(env.getGameState())
    state = torch.tensor(state, device=device).unsqueeze(0)

    while not env.game_over():
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            with torch.no_grad():
                action = int(torch.argmax(main_model(state)).item())

        reward = env.act(actions[action])
        reward = np.clip(reward, -1, 1)
        total_reward += reward

        next_state_np = normalize_state(env.getGameState())
        done = env.game_over()
        next_state = torch.tensor(next_state_np, device=device).unsqueeze(0)

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        step_count += 1

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions_batch, rewards_batch, next_states, dones = zip(*batch)

            states = torch.cat(states) 
            actions_batch = torch.tensor(actions_batch, device=device, dtype=torch.long)
            rewards_batch = torch.tensor(rewards_batch, device=device, dtype=torch.float32)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, device=device, dtype=torch.float32)

            predictions = main_model(states)
            targets = predictions.clone().detach()

            with torch.no_grad():
                next_q = target_model(next_states)
                max_next_q, _ = torch.max(next_q, dim=1)
                target_vals = rewards_batch + discount_factor * max_next_q * (1 - dones)

            for i in range(batch_size):
                targets[i, actions_batch[i]] = target_vals[i]

            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
            optimizer.step()
        if step_count % target_update_freq == 0:
            target_model.load_state_dict(main_model.state_dict())

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}: Total reward = {total_reward}, Epsilon = {epsilon:.4f}")
    episode_rewards.append(total_reward)

    if (episode + 1) % 500 == 0:
        torch.save(main_model.state_dict(), f"flappy_dqn_episode_{episode+1}.pth")

# Plot the episode rewards over time
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Deep-Q Rewards Per Episode')
plt.show()
