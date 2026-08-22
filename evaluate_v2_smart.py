import time
import numpy as np
import torch
from stable_baselines3 import PPO
from game2048_v2_env import Game2048V2Env

def get_best_valid_action(model, env, obs):
    """
    Sprawdza wszystkie 4 ruchy i wybiera ten o najwyższym 
    prawdopodobieństwie, który jest w 100% LEGALNY na planszy.
    """
    # Pobierz prawdopodobieństwa wszystkich 4 akcji z sieci neuronowej
    obs_tensor = torch.as_tensor(obs[None]).to(model.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]

    # Posortuj akcje od najbardziej do najmniej preferowanej
    ranked_actions = np.argsort(probs)[::-1]

    # Sprawdź, który z nich faktycznie zmienia planszę
    for action in ranked_actions:
        temp_env = Game2048V2Env()
        temp_env.board = env.board.copy()
        old_b = temp_env.board.copy()
        temp_env._move(action)
        if not np.array_equal(old_b, temp_env.board):
            return int(action)

    # Jeśli żaden ruch nie zmienia planszy -> koniec gry
    return int(ranked_actions[0])

def evaluate_smart(model_path="models/ppo_2048_cnn_v2", n_games=50):
    print(f"\n📂 Ładowanie modelu CNN V2 ze Smart Action Masking...")
    model = PPO.load(model_path)

    scores = []
    max_tiles = []
    move_counts = []

    print(f"🎮 Rozgrywam {n_games} gier z inteligentnym filtrowaniem ruchów...\n")

    for game_num in range(n_games):
        env = Game2048V2Env()
        obs, info = env.reset()
        done = False

        while not done:
            action = get_best_valid_action(model, env, obs)
            obs, reward, done, truncated, info = env.step(action)

        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        move_counts.append(info["move_count"])

        if (game_num + 1) % 10 == 0:
            print(f"   Ukończono {game_num + 1}/{n_games} gier...")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)
    move_counts = np.array(move_counts)

    print(f"\n{'='*55}")
    print(f"  🧠 WYNIKI INTELIGENTNEGO BOTA CNN V2 ({n_games} gier)")
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
    for tile in [64, 128, 256, 512, 1024, 2048]:
        count = np.sum(max_tiles >= tile)
        pct = count / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"     ≥ {tile:>5}: {count:>3}/{n_games} ({pct:>5.1f}%) {bar}")
    print(f"{'='*55}\n")

def watch_live_smart(model_path="models/ppo_2048_cnn_v2", delay=0.12):
    print("🎬 OGLĄDAMY MECZ INTELIGENTNEGO BOTA CNN!\n")
    model = PPO.load(model_path)
    env = Game2048V2Env()
    obs, info = env.reset()
    done = False
    step = 0

    env.render()
    time.sleep(0.8)

    while not done:
        action = get_best_valid_action(model, env, obs)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        print(f"Ruch #{step}: {env.action_names[action]}")
        env.render()
        time.sleep(delay)

    print(f"\n🏁 KONIEC ROZGRYWKI!")
    print(f"Końcowy Score: {info['score']:,} | Max kafelek: {info['max_tile']} | Ruchy: {info['move_count']}")

if __name__ == "__main__":
    evaluate_smart()
    input("\nNaciśnij [ENTER], aby obejrzeć inteligentny mecz na żywo...")
    watch_live_smart()
