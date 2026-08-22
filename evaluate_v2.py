import time
import numpy as np
from stable_baselines3 import PPO
from game2048_v2_env import Game2048V2Env

def evaluate(model_path="models/ppo_2048_cnn_v2", n_games=100):
    print(f"\n📂 Ładowanie modelu CNN V2: {model_path}...")
    model = PPO.load(model_path)

    scores = []
    max_tiles = []
    move_counts = []

    print(f"🎮 Rozgrywam {n_games} gier testowych...")

    for game_num in range(n_games):
        env = Game2048V2Env()
        obs, info = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))

        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        move_counts.append(info["move_count"])

        if (game_num + 1) % 20 == 0:
            print(f"   Ukończono {game_num + 1}/{n_games} gier...")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print(f"\n{'='*55}")
    print(f"  📊 WYNIKI MODELU CNN V2 ({n_games} gier)")
    print(f"{'='*55}")
    print(f"  🏆 PUNKTY (SCORE):")
    print(f"     Średni wynik:   {scores.mean():>10,.0f}")
    print(f"     Mediana:        {np.median(scores):>10,.0f}")
    print(f"     Rekord życiowy: {scores.max():>10,.0f}")
    print(f"")
    print(f"  🧱 NAJWYŻSZE KAFELKI:")
    print(f"     Średni max kafelek: {max_tiles.mean():>8,.0f}")
    print(f"     Najwyższy w historii: {max_tiles.max():>6,.0f}")
    print(f"")
    print(f"  📈 ROZKŁAD KAFELKÓW W %:")
    for tile in [128, 256, 512, 1024, 2048, 4096]:
        count = np.sum(max_tiles >= tile)
        pct = count / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"     ≥ {tile:>5}: {count:>3}/{n_games} ({pct:>5.1f}%) {bar}")
    print(f"{'='*55}\n")

def watch_live(model_path="models/ppo_2048_cnn_v2", delay=0.12):
    print("🎬 OGLĄDAMY MECZ NA ŻYWO (CNN V2)!\n")
    model = PPO.load(model_path)
    env = Game2048V2Env()
    obs, info = env.reset()
    done = False
    step = 0

    env.render()
    time.sleep(0.8)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        step += 1

        print(f"Ruch #{step}: {env.action_names[int(action)]}")
        env.render()
        time.sleep(delay)

    print(f"\n🏁 KONIEC ROZGRYWKI!")
    print(f"Finalny Score: {info['score']:,} | Max kafelek: {info['max_tile']}")

if __name__ == "__main__":
    evaluate()
    input("\nNaciśnij [ENTER], aby obejrzeć mecz na żywo...")
    watch_live()
