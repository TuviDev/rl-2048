import time
import numpy as np
from stable_baselines3 import PPO
from game2048_env import Game2048Env

def watch_live_game(model_path="models/ppo_2048_turbo", delay=0.12):
    print("🎬 OGLĄDAMY MECZ NA ŻYWO! (Każdy ruch co 0.12s)\n")
    model = PPO.load(model_path)
    env = Game2048Env()
    obs, info = env.reset()
    done = False
    step = 0

    env.render()
    time.sleep(0.8)

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        action_idx = int(action)  # Bezpieczna konwersja na int

        obs, reward, done, truncated, info = env.step(action_idx)
        step += 1

        print(f"Ruch #{step}: {env.action_names[action_idx]}")
        env.render()
        time.sleep(delay)

    print(f"\n🏁 KONIEC ROZGRYWKI!")
    print(f"Końcowy Score: {info['score']:,} | Max kafelek: {info['max_tile']} | Ruchy: {info['move_count']}")

if __name__ == "__main__":
    watch_live_game()
