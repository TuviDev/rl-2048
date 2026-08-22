import time
import numpy as np
from numba import njit

# Najpotężniejsza Matryca Węża V4
SNAKE_MATRIX = np.array([
    [1000000.0, 100000.0, 10000.0, 1000.0],
    [100.0,     200.0,    400.0,   800.0],
    [80.0,      60.0,     40.0,    20.0],
    [1.0,       2.0,      3.0,     4.0]
], dtype=np.float64)

@njit(fastmath=True)
def move_left(board):
    new_b = np.zeros((4, 4), dtype=np.int64)
    score = 0
    changed = False

    for r in range(4):
        row = board[r]
        non_zero = row[row != 0]
        merged = np.zeros(4, dtype=np.int64)
        idx = 0
        skip = False

        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                val = non_zero[i] * 2
                merged[idx] = val
                score += val
                skip = True
                idx += 1
            else:
                merged[idx] = non_zero[i]
                idx += 1

        for c in range(4):
            new_b[r, c] = merged[c]
            if board[r, c] != merged[c]:
                changed = True

    return new_b, score, changed

@njit(fastmath=True)
def simulate_move_numba(board, action):
    if action == 0:   # Left
        return move_left(board)
    elif action == 1: # Up
        b_rot = np.rot90(board, -1)
        b_new, s, ch = move_left(b_rot)
        return np.rot90(b_new, 1), s, ch
    elif action == 2: # Right
        b_rot = np.rot90(board, 2)
        b_new, s, ch = move_left(b_rot)
        return np.rot90(b_new, 2), s, ch
    elif action == 3: # Down
        b_rot = np.rot90(board, 1)
        b_new, s, ch = move_left(b_rot)
        return np.rot90(b_new, -1), s, ch
    return board, 0, False

@njit(fastmath=True)
def evaluate_single_aspect(board):
    score = 0.0
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            if val > 0:
                score += val * SNAKE_MATRIX[r, c]

    empty = 0
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                empty += 1
    score += empty * 50000.0
    return score

@njit(fastmath=True)
def evaluate_board_d4_symmetry(board):
    """Ocena planszy uwzględniająca pełną 8-krotną symetrię grupy D4 (4 obroty + odbicie)."""
    max_eval = -1e18
    b_curr = board.copy()

    for rot in range(4):
        e1 = evaluate_single_aspect(b_curr)
        if e1 > max_eval:
            max_eval = e1
        b_flip = np.fliplr(b_curr)
        e2 = evaluate_single_aspect(b_flip)
        if e2 > max_eval:
            max_eval = e2
        b_curr = np.rot90(b_curr, 1)

    return max_eval

@njit(fastmath=True)
def expectimax_v4(board, depth, is_player):
    if depth == 0:
        return evaluate_board_d4_symmetry(board)

    if is_player:
        best_score = -1e18
        moved = False
        for a in range(4):
            b_next, _, changed = simulate_move_numba(board, a)
            if changed:
                moved = True
                score = expectimax_v4(b_next, depth - 1, False)
                if score > best_score:
                    best_score = score
        return best_score if moved else -1e12
    else:
        empty_positions = []
        for r in range(4):
            for c in range(4):
                if board[r, c] == 0:
                    empty_positions.append((r, c))

        n_empty = len(empty_positions)
        if n_empty == 0:
            return evaluate_board_d4_symmetry(board)

        if n_empty > 3 and depth >= 3:
            empty_positions = empty_positions[:3]
            n_empty = 3

        expected = 0.0
        for pos in empty_positions:
            r, c = pos[0], pos[1]
            b2 = board.copy()
            b2[r, c] = 2
            expected += 0.9 * expectimax_v4(b2, depth - 1, True)

            b4 = board.copy()
            b4[r, c] = 4
            expected += 0.1 * expectimax_v4(b4, depth - 1, True)

        return expected / float(n_empty)

def get_adaptive_god_move(board, valid_mask=None):
    board_64 = board.astype(np.int64)
    empty_tiles = np.sum(board == 0)

    # DYNAMIC DEPTH SCALING
    if empty_tiles <= 3:
        target_depth = 5
    else:
        target_depth = 4

    best_action = -1
    best_score = -1e18

    for a in range(4):
        if valid_mask is not None and not valid_mask[a]:
            continue

        b_next, _, changed = simulate_move_numba(board_64, a)
        if changed:
            score = expectimax_v4(b_next, target_depth - 1, False)
            if score > best_score:
                best_score = score
                best_action = a

    if best_action == -1:
        if valid_mask is not None and np.any(valid_mask):
            best_action = int(np.where(valid_mask)[0][0])
        else:
            best_action = 0

    return best_action, target_depth
