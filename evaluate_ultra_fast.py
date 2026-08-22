import time
import numpy as np
from game2048_v2_env import Game2048V2Env

# Matryca wagi węża — układa kafelki od największego w rogu do najmniejszego
SNAKE_MATRIX = np.array([
    [15, 14, 13, 12],
    [8,   9, 10, 11],
    [7,   6,  5,  4],
    [0,   1,  2,  3]
], dtype=np.float32)

def evaluate_board_fast(board):
    """Błyskawiczna ocena planszy w ułamku milisekundy za pomocą operacji macierzowych."""
    # Oblicz log2 dla kafelków
    log_board = np.zeros_like(board, dtype=np.float32)
    mask = board > 0
    log_board[mask] = np.log2(board[mask])

    # 1. Punkty za ułożenie w węża
    snake_score = np.sum(log_board * SNAKE_MATRIX)

    # 2. Bonus za puste pola
    empty_bonus = np.sum(board == 0) * 10.0

    # 3. Bonus jeśli największy kafelek jest w lewym-górnym rogu
    max_tile = np.max(board)
    corner_bonus = 50.0 if board[0, 0] == max_tile else -50.0

    return float(snake_score + empty_bonus + corner_bonus)

def expectimax(board, depth, is_player):
    if depth == 0:
        return evaluate_board_fast(board)

    if is_player:
        best_score = -float('inf')
        moved = False
        for action in range(4):
            temp_env = Game2048V2Env()
            temp_env.board = board.copy()
            old_b = temp_env.board.copy()
            temp_env._move(action)
            if not np.array_equal(old_b, temp_env.board):
                moved = True
                score = expectimax(temp_env.board, depth - 1, is_player=False)
                best_score = max(best_score, score)
        return best_score if moved else -10000.0
    else:
        empty_positions = list(zip(*np.where(board == 0)))
        if not empty_positions:
            return evaluate_board_fast(board)

        # Ograniczenie do max 3 losowych pól dla płynności
        if len(empty_positions) > 3:
            import random
            empty_positions = random.sample(empty_positions, 3)

        expected_score = 0.0
        for pos in empty_positions:
            b2 = board.copy()
            b2[pos] = 2
            expected_score += 0.9 * expectimax(b2, depth - 1, is_player=True)

            b4 = board.copy()
            b4[pos] = 4
            expected_score += 0.1 * expectimax(b4, depth - 1, is_player=True)

        return expected_score / len(empty_positions)

def get_best_move(board, depth=3):
    best_action = 0
    best_score = -float('inf')

    for action in range(4):
        temp_env = Game2048V2Env()
        temp_env.board = board.copy()
        old_b = temp_env.board.copy()
        temp_env._move(action)
        if not np.array_equal(old_b, temp_env.board):
            score = expectimax(temp_env.board, depth - 1, is_player=False)
            if score > best_score:
                best_score = score
                best_action = action

    return best_action

def evaluate_ultra(n_games=3, depth=3):
    scores = []
    max_tiles = []

    print(f"\n⚡ ROZPOZCZYNAM SZYBKĄ EWALUACJĘ AI (Depth={depth})...\n")

    for g in range(n_games):
        env = Game2048V2Env()
        obs, info = env.reset()
        done = False

        while not done:
            action = get_best_move(env.board, depth=depth)
            obs, reward, done, truncated, info = env.step(action)

        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        print(f"  Gra #{g+1:>2}: Score: {info['score']:>8,} | Max Kafelek: {info['max_tile']:>5} | Ruchy: {info['move_count']:>4}")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print(f"\n{'='*55}")
    print(f"  🏆 WYNIKI ULTRA-SZYBKIEGO MISTRZA DEPTH={depth}")
    print(f"{'='*55}")
    print(f"  Średni wynik:   {scores.mean():>10,.0f}")
    print(f"  Rekord:         {scores.max():>10,.0f}")
    print(f"  Średni kafelek: {max_tiles.mean():>10,.0f}")
    print(f"  Najwyższy:      {max_tiles.max():>10,.0f}")
    print(f"")
    print(f"  📈 ROZKŁAD KAFELKÓW:")
    for tile in [256, 512, 1024, 2048]:
        count = np.sum(max_tiles >= tile)
        pct = count / n_games * 100
        bar = "█" * int(pct / 2)
        print(f"     ≥ {tile:>5}: {count:>2}/{n_games} ({pct:>5.1f}%) {bar}")
    print(f"{'='*55}\n")

def watch_live_ultra(depth=3, delay=0.08):
    env = Game2048V2Env()
    obs, info = env.reset()
    done = False
    step = 0

    print(f"🎬 OGLĄDAMY BŁYSKAWICZNY MECZ NA ŻYWO (Depth={depth})!\n")
    env.render()
    time.sleep(1)

    while not done:
        action = get_best_move(env.board, depth=depth)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        print(f"Ruch #{step}: {env.action_names[action]}")
        env.render()
        time.sleep(delay)

    print(f"\n🏁 KONIEC ROZGRYWKI!")
    print(f"Finalny Score: {info['score']:,} | Max kafelek: {info['max_tile']}")

if __name__ == "__main__":
    evaluate_ultra(n_games=3, depth=3)
    input("\nNaciśnij [ENTER], aby obejrzeć ultraszybki mecz na żywo...")
    watch_live_ultra(depth=3)
