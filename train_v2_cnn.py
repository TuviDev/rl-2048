import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from game2048_v2_env import Game2048V2Env

# Dedykowana architektura CNN dla planszy 2048 (16x4x4)
class BoardCNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # Wejście: (16 kanałów, 4x4)
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def train_v2():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    NUM_ENVS = 16
    print(f"🔥 Odpalam {NUM_ENVS} instancji One-Hot na 16 wątkach Xeona...")
    env = make_vec_env(Game2048V2Env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Obliczenia konwolucyjne CNN na GPU: {device.upper()} ({torch.cuda.get_device_name(0)})")

    policy_kwargs = dict(
        features_extractor_class=BoardCNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard_v2",
        device=device,
        verbose=1,
    )

    total_steps = 2_000_000
    print(f"\n⚡ Rozpoczynam trening CNN V2 na {total_steps:,} krokach (~15 min)...")

    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
    )

    model_path = "models/ppo_2048_cnn_v2"
    model.save(model_path)
    print(f"\n✅ SUPERMODEL CNN V2 ZAPISANY: {model_path}")

if __name__ == "__main__":
    train_v2()
