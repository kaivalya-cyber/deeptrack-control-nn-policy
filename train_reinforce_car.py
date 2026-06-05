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
    """CNN policy for image obs: outputs mean/std for actions and state value."""

    def __init__(self, in_channels: int = 4, action_space_dims: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
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
        self.value_net = nn.Linear(256, 1)
        self._action_dims = action_space_dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, in_channels, 96, 96)
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        action_means = self.policy_mean_net(features)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(features)))
        state_value = self.value_net(features)
        return action_means, action_stddevs, state_value


class ActorCritic:
    """Actor-Critic agent (upgraded from REINFORCE)."""

    def __init__(
        self,
        policy: nn.Module,
        action_space_dims: int,
        learning_rate: float = 1e-4,
        entropy_coeff: float = 0.01,
        critic_coeff: float = 0.5,
    ):
        self.gamma = 0.99
        self.eps = 1e-6
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff

        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.entropies: list[torch.Tensor] = []

        self.net = policy
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate)
        self._action_dims = action_space_dims

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        # state: (4, 96, 96) uint8 -> (1, 4, 96, 96) float
        x = torch.tensor(np.asarray(state, dtype=np.float32) / 255.0).unsqueeze(0)
        action_means, action_stddevs, state_value = self.net(x)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        log_prob = distrib.log_prob(action).sum()
        entropy = distrib.entropy().sum()

        self.log_probs.append(log_prob)
        self.values.append(state_value.squeeze())
        self.entropies.append(entropy)

        return action.detach().numpy()

    def update(self) -> None:
        running_g = 0.0
        returns: list[float] = []
        for r in self.rewards[::-1]:
            running_g = r + self.gamma * running_g
            returns.insert(0, running_g)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        log_probs_t = torch.stack(self.log_probs)
        values_t = torch.stack(self.values)
        entropies_t = torch.stack(self.entropies)

        advantages = returns_t - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = nn.functional.mse_loss(values_t, returns_t)
        entropy_loss = -entropies_t.mean()

        loss = actor_loss + self.critic_coeff * critic_loss + self.entropy_coeff * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []


def main() -> None:
    parser = argparse.ArgumentParser(description="Actor-Critic on CarRacing-v3 (car racing around track)")
    parser.add_argument("--episodes", type=int, default=int(2e3), help="Episodes per seed (default: 2000)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument("--no-show", action="store_true", help="Only save plot, do not show")
    parser.add_argument("--render", action="store_true", help="Render the car racing environment during training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--critic", type=float, default=0.5, help="Critic loss weight (default: 0.5)")
    parser.add_argument("--load", type=str, default=None, help="Path to load a saved model")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    if args.render:
        print("Rendering enabled: a CarRacing window should open. Close it or wait for training to finish.")
    def make_env(seed: int | None = None) -> gym.Env:
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = gym.wrappers.RecordEpisodeStatistics(env, 50)
        if seed is not None:
            env.reset(seed=seed)
        return env

    base_env = make_env()
    action_space_dims = base_env.action_space.shape[0]  # 3
    base_env.close()

    total_num_episodes = args.episodes
    seed_list = [1, 2, 3, 5, 8]
    seeds = seed_list[: args.seeds]
    rewards_over_seeds: list[list[float]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        env = make_env(seed=seed)
        policy = CNNPolicy(in_channels=4, action_space_dims=action_space_dims)
        if args.load and os.path.exists(args.load):
            print(f"Loading model from {args.load}")
            policy.load_state_dict(torch.load(args.load))

        agent = ActorCritic(
            policy,
            action_space_dims,
            learning_rate=args.lr,
            entropy_coeff=args.entropy,
            critic_coeff=args.critic,
        )
        reward_over_episodes: list[float] = []
        best_avg_reward = -float("inf")

        for episode in range(total_num_episodes):
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                action = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                agent.rewards.append(reward)
                done = terminated or truncated

            reward_over_episodes.append(env.return_queue[-1])
            agent.update()

            avg_reward = np.mean(env.return_queue)
            if avg_reward > best_avg_reward and episode > 10:
                best_avg_reward = avg_reward
                torch.save(agent.net.state_dict(), os.path.join(SCRIPT_DIR, f"best_model_seed_{seed}.pt"))

            print_interval = max(1, total_num_episodes // 10)
            if episode % print_interval == 0:
                print(f"Seed {seed} | Episode: {episode} | Average Reward: {int(avg_reward)}")

        rewards_over_seeds.append(reward_over_episodes)
        env.close()

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
