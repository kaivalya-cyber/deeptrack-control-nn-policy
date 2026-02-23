"""
REINFORCE for Car Racing (Box2D).
Separate simulation: car racing around a track. Uses Gymnasium CarRacing-v3
with a CNN policy for image observations. Same REINFORCE + learning-curve approach
as the Hopper track project.
"""
from __future__ import annotations

import argparse
import os
import random

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

plt.rcParams["figure.figsize"] = (10, 5)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class CNNPolicy(nn.Module):
    """CNN policy for image obs (96x96x3): outputs mean and std for 3 actions (steer, gas, brake)."""

    def __init__(self, action_space_dims: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.Tanh(),
        )
        # 96 -> 23 -> 10 -> 8; 8*8*32 = 2048
        self.flatten_size = 32 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.Tanh(),
        )
        self.policy_mean_net = nn.Linear(256, action_space_dims)
        self.policy_stddev_net = nn.Linear(256, action_space_dims)
        self._action_dims = action_space_dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 3, 96, 96)
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        action_means = self.policy_mean_net(features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(features)))
        return action_means, action_stddevs


class REINFORCE:
    """REINFORCE algorithm (same as Hopper track)."""

    def __init__(self, policy: nn.Module, action_space_dims: int):
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6
        self.probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.net = policy
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self._action_dims = action_space_dims

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        # state: (96, 96, 3) uint8 -> (1, 3, 96, 96) float
        x = np.asarray(state, dtype=np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        x = torch.tensor(x[np.newaxis, ...])
        action_means, action_stddevs = self.net(x)
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action).sum()
        self.probs.append(prob)
        return action.numpy()

    def update(self) -> None:
        running_g = 0.0
        gs: list[float] = []
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)
        deltas = torch.tensor(gs, dtype=torch.float32)
        log_probs = torch.stack(self.probs).squeeze()
        loss = -torch.sum(log_probs * deltas)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.probs = []
        self.rewards = []


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE on CarRacing-v3 (car racing around track)")
    parser.add_argument("--episodes", type=int, default=int(2e3), help="Episodes per seed (default: 2000)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument("--no-show", action="store_true", help="Only save plot, do not show")
    parser.add_argument("--render", action="store_true", help="Render the car racing environment during training")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    if args.render:
        print("Rendering enabled: a CarRacing window should open. Close it or wait for training to finish.")
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

    action_space_dims = env.action_space.shape[0]  # 3
    policy = CNNPolicy(action_space_dims=action_space_dims)
    agent = REINFORCE(policy, action_space_dims)

    total_num_episodes = args.episodes
    seed_list = [1, 2, 3, 5, 8]
    seeds = seed_list[: args.seeds]
    rewards_over_seeds: list[list[float]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        agent = REINFORCE(CNNPolicy(action_space_dims=action_space_dims), action_space_dims)
        reward_over_episodes: list[float] = []

        for episode in range(total_num_episodes):
            obs, info = wrapped_env.reset(seed=seed)
            done = False
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)
                done = terminated or truncated

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            print_interval = max(1, total_num_episodes // 10)
            if episode % print_interval == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print(f"Seed {seed} | Episode: {episode} | Average Reward: {avg_reward}")

        rewards_over_seeds.append(reward_over_episodes)

    wrapped_env.close()

    n = len(rewards_over_seeds[0])
    window = max(1, n // 10)
    for i, seed in enumerate(seeds):
        early = np.mean(rewards_over_seeds[i][:window])
        late = np.mean(rewards_over_seeds[i][-window:])
        print(f"Seed {seed}: early avg reward = {early:.1f}, late avg reward = {late:.1f} (learning: {late > early})")

    df = pd.DataFrame(rewards_over_seeds).melt()
    df.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df).set(title="REINFORCE on CarRacing-v3")
    out_path = os.path.join(SCRIPT_DIR, "learning_curve_car.png")
    plt.savefig(out_path)
    print(f"Saved learning curve to {out_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
