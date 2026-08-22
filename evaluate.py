import time
import numpy as np
from game2048_v3_mask import Game2048V3MaskEnv
from expectimax_god_engine import get_god_move, clear_transposition_table

def evaluate_main(n_games=3):
    print("\n" + "="*60)
    print("👑 EVALUATING BEST WORKING ENGINE (Expectimax D4 + Bitboard Cache)")
    print("="*60 + "\n")

    dummy_b = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [0, 0, 2, 4]], dtype=np.int32)
    get_god_move(dummy_b)
    print("✅ C++ Numba Engine compiled successfully!\n")

    scores = []
    max_tiles = []

    for g in range(n_games):
        clear_transposition_table()
        env = Game2048V3MaskEnv()
        obs, info = env.reset()
        done = False
        start_time = time.time()

        while not done:
            masks = env.action_masks()
            action, depth_used = get_god_move(env.board, valid_mask=masks)
            obs, reward, done, truncated, info = env.step(action)

        elapsed = time.time() - start_time
        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        medal = "🥇" if info["max_tile"] >= 4096 else ("🥈" if info["max_tile"] >= 2048 else "🥉")
        print(f"  {medal} Game #{g+1}: Score = {info['score']:>8,} | Max Tile = {info['max_tile']:>5} | Time = {elapsed:.1f}s | Moves = {info['move_count']}")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print("\n" + "="*60)
    print(f"  📊 BENCHMARK SUMMARY ({n_games} Games)")
    print("="*60)
    print(f"  Average Score: {scores.mean():>10,.0f} pts")
    print(f"  Max Score:     {scores.max():>10,.0f} pts")
    print(f"  Max Tile:      {max_tiles.max():>10}")
    print("="*60 + "\n")

if __name__ == "__main__":
    evaluate_main(n_games=3)
