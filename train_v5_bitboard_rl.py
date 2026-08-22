import os
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from game2048_v5_bitenv import Game2048BitboardEnv

class DeepBitboardCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        # Głębsza sieć splotowa z Residual-like Layers dla Bitboardu
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=1),
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

def train_v5_rl():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    NUM_ENVS = 16
    print(f"🔥 Odpalam MASKABLE PPO 10,000,000 kroków na BITBOARD O(1) + RTX 3070...")
    env = make_vec_env(Game2048BitboardEnv, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Sprzętowe wspomaganie CUDA: {device.upper()} ({torch.cuda.get_device_name(0)})")

    policy_kwargs = dict(
        features_extractor_class=DeepBitboardCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
    )

    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard_v5_bitrl",
        device=device,
        verbose=1,
    )

    total_steps = 10_000_000
    print(f"\n⚡ Rozpoczynam trening MaskablePPO na {total_steps:,} krokach (~4 min)...")

    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
    )

    model_path = "models/maskable_ppo_v5_10m"
    model.save(model_path)
    print(f"\n✅ APOSTOLSKI MODEL V5 10M ZAPISANY: {model_path}")

if __name__ == "__main__":
    train_v5_rl()
