import time
import numpy as np
from game2048_v3_mask import Game2048V3MaskEnv
from bitboard_v5_god import get_v5_ultimate_move

def run_v5_tournament(n_games=3):
    print(f"\n👑 ROZPOZCZYNAM TURNIEJ ULTIMATE BITBOARD V5 (O(1) LOOKUP, BEZPIECZNE PĘTLE)...\n")

    # Warmup Numba JIT
    dummy_b = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [0, 0, 2, 4]], dtype=np.int32)
    get_v5_ultimate_move(dummy_b)
    print("✅ Kompilacja Bitboard O(1) C++ zakończona sukcesem!\n")

    scores = []
    max_tiles = []

    for g in range(n_games):
        env = Game2048V3MaskEnv()
        obs, info = env.reset()
        done = False
        start_time = time.time()

        while not done:
            masks = env.action_masks()
            action, depth_used = get_v5_ultimate_move(env.board, valid_mask=masks)
            obs, reward, done, truncated, info = env.step(action)

        elapsed = time.time() - start_time
        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        print(f"  🏆 Gra #{g+1}: Wynik = {info['score']:>9,} | Max Kafelek = {info['max_tile']:>5} | Czas = {elapsed:.1f}s | Ruchy = {info['move_count']}")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print(f"\n{'='*65}")
    print(f"  👑 OSTATECZNE WYNIKI SILNIKA V5 BITBOARD")
    print(f"{'='*65}")
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
    print(f"{'='*65}\n")

if __name__ == "__main__":
    run_v5_tournament(n_games=3)
