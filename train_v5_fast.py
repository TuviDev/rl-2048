import os
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from game2048_v5_bitenv_fast import Game2048BitboardEnvFast

# Lekka i superszybka sieć CNN (Bez spowalniającego BatchNorm!)
class UltraFastCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
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

def train_v5_fast():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 24 RÓWNOLEGŁE ŚRODOWISKA DLA 28 WĄTKÓW XEONA!
    NUM_ENVS = 24
    print(f"🔥 Odpalam ULTRA FAST MASKABLE PPO ({NUM_ENVS} środowisk) na RTX 3070...")
    env = make_vec_env(Game2048BitboardEnvFast, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Urządzenie: {device.upper()} ({torch.cuda.get_device_name(0)})")

    policy_kwargs = dict(
        features_extractor_class=UltraFastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = MaskablePPO(
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
        tensorboard_log="./logs/tensorboard_v5_fast",
        device=device,
        verbose=1,
    )

    total_steps = 10_000_000
    print(f"\n⚡ Rozpoczynam TURBO trening na {total_steps:,} krokach...")

    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
    )

    model_path = "models/maskable_ppo_v5_10m"
    model.save(model_path)
    print(f"\n✅ APOSTOLSKI MODEL V5 ZAPISANY: {model_path}")

if __name__ == "__main__":
    train_v5_fast()
