import time
import numpy as np
from game2048_v3_mask import Game2048V3MaskEnv
from expectimax_god_v4 import get_adaptive_god_move

def run_v4_tournament(n_games=3):
    print(f"\n👑 ROZPOZCZYNAM TURNIEJ GOD MODE V4 (D4 Symmetry + Adaptive Depth scaling)...\n")

    # Warmup Numba JIT
    dummy_b = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64)
    get_adaptive_god_move(dummy_b)
    print("✅ Kompilator Numba C++ z rozszerzeniem D4 Symmetry gotowy!\n")

    scores = []
    max_tiles = []

    for g in range(n_games):
        env = Game2048V3MaskEnv()
        obs, info = env.reset()
        done = False
        start_time = time.time()

        while not done:
            action, depth_used = get_adaptive_god_move(env.board)
            obs, reward, done, truncated, info = env.step(action)

        elapsed = time.time() - start_time
        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        print(f"  🏆 Gra #{g+1}: Score = {info['score']:>8,} | Max Kafelek = {info['max_tile']:>5} | Czas = {elapsed:.1f}s | Ruchy = {info['move_count']}")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print(f"\n{'='*60}")
    print(f"  👑 OSTATECZNE WYNIKI SILNIKA GOD MODE V4")
    print(f"{'='*60}")
    print(f"  Średni wynik:         {scores.mean():>10,.0f}")
    print(f"  Rekord punktowy:      {scores.max():>10,.0f}")
    print(f"  Średni max kafelek:   {max_tiles.mean():>10,.0f}")
    print(f"  Najwyższy kafelek:    {max_tiles.max():>10,.0f}")
    print(f"")
    print(f"  📈 ROZKŁAD KAFELKÓW:")
    for tile in [512, 1024, 2048, 4096]:
        count = np.sum(max_tiles >= tile)
        pct = count / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"     ≥ {tile:>5}: {count:>2}/{n_games} ({pct:>5.1f}%) {bar}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_v4_tournament(n_games=3)
