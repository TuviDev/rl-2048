import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from game2048_env import Game2048Env

def train_turbo():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    NUM_ENVS = 16
    print(f"🔥 Uruchamiam {NUM_ENVS} równoległych gier na procesorze Xeon...")
    env = make_vec_env(Game2048Env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Obliczenia na: {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else ''})")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log="./logs/tensorboard_turbo",
        device=device,
        verbose=1,
    )

    total_steps = 500_000
    print(f"\n⚡ Rozpoczynam czysty trening na {total_steps:,} krokach...")
    
    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
    )

    model_path = "models/ppo_2048_turbo"
    model.save(model_path)
    print(f"\n✅ Supermodel zapisany jako: {model_path}")

if __name__ == "__main__":
    train_turbo()
