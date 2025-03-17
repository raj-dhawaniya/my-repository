import gymnasium as gym
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN training function
def dqn(num_episodes=1000, batch_size=64, gamma=0.99, epsilon=0.1, lr=0.001):
    env = gym.make('LunarLander-v3')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    q_network = QNetwork(input_size, output_size)
    target_network = QNetwork(input_size, output_size)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    replay_buffer = deque(maxlen=10000)

    best_reward = -np.inf
    best_params = None

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(torch.tensor(observation, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            # Take action and observe next state and reward
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience in replay buffer
            replay_buffer.append((observation, action, reward, next_observation, done))

            observation = next_observation
            episode_reward += reward

            # Train the network using experience replay
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Compute Q-values and target Q-values
                q_values = q_network(states).gather(1, actions.unsqueeze(1))
                next_q_values = target_network(next_states).max(1)[0].detach()
                targets = rewards + gamma * next_q_values * (1 - dones)

                # Compute loss and update the network
                loss = nn.MSELoss()(q_values.squeeze(), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        env.close()

        # Update best parameters
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_params = q_network.state_dict()

        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

    return best_params

# Train and save the best policy
def train_and_save(filename, num_episodes=1000):
    best_params = dqn(num_episodes=num_episodes)
    torch.save(best_params, filename)
    print(f"Best policy saved to {filename}")

# Load the trained policy
def load_policy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = torch.load(filename)
    print(f"Loaded best policy from {filename}")
    return best_params

# Play with the trained policy
def play_policy(best_params, episodes=5):
    env = gym.make('LunarLander-v3', render_mode='human')
    q_network = QNetwork(8, 4)
    q_network.load_state_dict(best_params)

    total_reward = 0.0
    for _ in range(episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                q_values = q_network(torch.tensor(observation, dtype=torch.float32))
                action = torch.argmax(q_values).item()

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_reward += episode_reward
        print(f"Episode reward: {episode_reward:.2f}")

    env.close()
    print(f"Average reward over {episodes} episodes: {total_reward / episodes:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using DQN.")
    parser.add_argument("--train", action="store_true", help="Train the policy using DQN and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.pth", help="Filename to save/load the best policy.")
    args = parser.parse_args()

    if args.train:
        # Train and save the best policy
        train_and_save(args.filename, num_episodes=1000)
    elif args.play:
        # Load and play with the best policy
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")