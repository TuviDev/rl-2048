import time
import numpy as np
from game2048_v3_mask import Game2048V3MaskEnv
from bitboard_v5_god import execute_move_bitboard, expectimax_v5, numpy_to_bitboard, get_v5_ultimate_move

def inspect_v5_decisions(board_np, depth=4):
    """Pokazuje ocenę punktową każdego z 4 ruchów wyliczoną przez V5."""
    b64 = numpy_to_bitboard(board_np)
    evals = {}

    for a, a_name in enumerate(["← LEWO", "↑ GÓRA", "→ PRAWO", "↓ DÓŁ"]):
        b_next, score_gain, changed = execute_move_bitboard(b64, a)
        if changed:
            eval_score = expectimax_v5(b_next, depth - 1, False)
            evals[a_name] = eval_score
        else:
            evals[a_name] = "ZABLOKOWANY (Nielegalny)"

    return evals

def run_live_diagnostics(delay=0.2):
    print("\n" + "="*60)
    print("🎬 MOCNA DIAGNOSTYKA V5 NA ŻYWO: PODGLĄD DECYZJI AI KROK PO KROKU")
    print("="*60 + "\n")

    env = Game2048V3MaskEnv()
    obs, info = env.reset()
    done = False
    step = 0

    # Rozgrzewka kompilatora
    dummy_b = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    get_v5_ultimate_move(dummy_b)

    while not done:
        step += 1
        masks = env.action_masks()
        action, depth_used = get_v5_ultimate_move(env.board, valid_mask=masks)

        # Pobierz oceny wszystkich ruchów
        eval_details = inspect_v5_decisions(env.board, depth=depth_used)

        print(f"\n─────────────────────────────────────────────────────────────")
        print(f"RUCH #{step} | Max Kafelek: {env.max_tile} | Score: {env.score:,} | Depth={depth_used}")
        print(f"─────────────────────────────────────────────────────────────")
        env.render()

        print("📊 Ocena możliwości przez V5 Bitboard:")
        for a_name, score_val in eval_details.items():
            chosen_mark = "👈 WYBRANY" if a_name == env.action_names[action] else ""
            if isinstance(score_val, str):
                print(f"   {a_name:<10}: {score_val} {chosen_mark}")
            else:
                print(f"   {a_name:<10}: {score_val:>18,.1f} pkt {chosen_mark}")

        obs, reward, done, truncated, info = env.step(action)
        time.sleep(delay)

        # Jeśli gra nagle się kończy — pauza do analizy
        if done:
            print("\n" + "="*60)
            print(f"🏁 KONIEC ROZGRYWKI W {step}. RUCHU!")
            print(f"Ostateczny wynik: {info['score']:,} | Max kafelek: {info['max_tile']}")
            print("="*60)

if __name__ == "__main__":
    run_live_diagnostics(delay=0.15)
