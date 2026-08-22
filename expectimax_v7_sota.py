import time
import numpy as np
from numba import njit, uint16, uint32, uint64, int64, int8, float64, boolean, types

# === 1. PREKOMPILACJA TABLIC LOOKUP O(1) DLA RUCHÓW BITOWYCH ===

def build_row_tables():
    row_left_table = np.zeros(65536, dtype=np.uint64)
    row_right_table = np.zeros(65536, dtype=np.uint64)
    row_score_table = np.zeros(65536, dtype=np.uint64)

    for row_val in range(65536):
        cells = [
            (row_val >> 0) & 0x0F,
            (row_val >> 4) & 0x0F,
            (row_val >> 8) & 0x0F,
            (row_val >> 12) & 0x0F
        ]

        # LEWO
        non_zero = [c for c in cells if c > 0]
        merged_left = []
        score = 0
        skip = False

        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                new_power = non_zero[i] + 1
                merged_left.append(new_power)
                score += (1 << new_power)
                skip = True
            else:
                merged_left.append(non_zero[i])

        while len(merged_left) < 4:
            merged_left.append(0)

        left_val = np.uint64(merged_left[0] | (merged_left[1] << 4) | (merged_left[2] << 8) | (merged_left[3] << 12))
        row_left_table[row_val] = left_val
        row_score_table[row_val] = np.uint64(score)

        # PRAWO
        non_zero_rev = [c for c in reversed(cells) if c > 0]
        merged_right_temp = []
        skip = False

        for i in range(len(non_zero_rev)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero_rev) and non_zero_rev[i] == non_zero_rev[i + 1]:
                new_power = non_zero_rev[i] + 1
                merged_right_temp.append(new_power)
                skip = True
            else:
                merged_right_temp.append(non_zero_rev[i])

        while len(merged_right_temp) < 4:
            merged_right_temp.append(0)

        merged_right = merged_right_temp[::-1]
        right_val = np.uint64(merged_right[0] | (merged_right[1] << 4) | (merged_right[2] << 8) | (merged_right[3] << 12))
        row_right_table[row_val] = right_val

    return row_left_table, row_right_table, row_score_table

ROW_LEFT_TABLE, ROW_RIGHT_TABLE, ROW_SCORE_TABLE = build_row_tables()

# === 2. MODYFIKOWALNA TABLICA TRANSPOSZYCJI CACHE O(1) ===

TABLE_SIZE = 2000003
HASH_BOARD = np.zeros(TABLE_SIZE, dtype=np.uint64)
HASH_DEPTH = np.zeros(TABLE_SIZE, dtype=np.int8)
HASH_SCORE = np.zeros(TABLE_SIZE, dtype=np.float64)

def clear_transposition_table():
    global HASH_BOARD, HASH_DEPTH, HASH_SCORE
    HASH_BOARD.fill(0)
    HASH_DEPTH.fill(0)
    HASH_SCORE.fill(0.0)

# === 3. OPERACJE BITOWE O(1) ===

@njit(uint64(uint64), fastmath=True)
def transpose_bitboard(board):
    r0 = (board >> 0) & 0xFFFF
    r1 = (board >> 16) & 0xFFFF
    r2 = (board >> 32) & 0xFFFF
    r3 = (board >> 48) & 0xFFFF

    c0 = (r0 & 0x0F) | ((r1 & 0x0F) << 4) | ((r2 & 0x0F) << 8) | ((r3 & 0x0F) << 12)
    c1 = ((r0 >> 4) & 0x0F) | (((r1 >> 4) & 0x0F) << 4) | (((r2 >> 4) & 0x0F) << 8) | (((r3 >> 4) & 0x0F) << 12)
    c2 = ((r0 >> 8) & 0x0F) | (((r1 >> 8) & 0x0F) << 4) | (((r2 >> 8) & 0x0F) << 8) | (((r3 >> 8) & 0x0F) << 12)
    c3 = ((r0 >> 12) & 0x0F) | (((r1 >> 12) & 0x0F) << 4) | (((r2 >> 12) & 0x0F) << 8) | (((r3 >> 12) & 0x0F) << 12)

    return np.uint64(c0) | (np.uint64(c1) << 16) | (np.uint64(c2) << 32) | (np.uint64(c3) << 48)

@njit(types.Tuple((uint64, uint64, boolean))(uint64, int64), fastmath=True)
def execute_move_bitboard(board, action):
    if action == 1 or action == 3:
        b = transpose_bitboard(board)
    else:
        b = board

    r0 = int((b >> 0) & 0xFFFF)
    r1 = int((b >> 16) & 0xFFFF)
    r2 = int((b >> 32) & 0xFFFF)
    r3 = int((b >> 48) & 0xFFFF)

    if action == 0 or action == 1:
        nr0 = ROW_LEFT_TABLE[r0]
        nr1 = ROW_LEFT_TABLE[r1]
        nr2 = ROW_LEFT_TABLE[r2]
        nr3 = ROW_LEFT_TABLE[r3]
    else:
        nr0 = ROW_RIGHT_TABLE[r0]
        nr1 = ROW_RIGHT_TABLE[r1]
        nr2 = ROW_RIGHT_TABLE[r2]
        nr3 = ROW_RIGHT_TABLE[r3]

    new_b = np.uint64(nr0) | (np.uint64(nr1) << 16) | (np.uint64(nr2) << 32) | (np.uint64(nr3) << 48)

    if action == 1 or action == 3:
        new_b = transpose_bitboard(new_b)

    score = ROW_SCORE_TABLE[r0] + ROW_SCORE_TABLE[r1] + ROW_SCORE_TABLE[r2] + ROW_SCORE_TABLE[r3]
    changed = (new_b != board)

    return new_b, score, changed

# === 4. WIELOCZYNNIKOWA HEURYSTYKA SOTA V7 (SNAKE + SMOOTHNESS + MONOTONICITY) ===

SNAKE_MATRIX = np.array([
    [1000000.0, 100000.0, 10000.0, 1000.0],
    [100.0,     200.0,    400.0,   800.0],
    [80.0,      60.0,     40.0,    20.0],
    [1.0,       2.0,      3.0,     4.0]
], dtype=np.float64)

@njit(int64[:, :](uint64), fastmath=True)
def bitboard_to_numpy(board64):
    res = np.zeros((4, 4), dtype=np.int64)
    for r in range(4):
        for c in range(4):
            shift = (r * 4 + c) * 4
            power = int((board64 >> shift) & np.uint64(0x0F))
            if power > 0:
                res[r, c] = 1 << power
    return res

@njit(float64(int64[:, :]), fastmath=True)
def evaluate_sota_v7_aspect(board):
    snake_score = 0.0
    empty_count = 0
    smoothness = 0.0
    max_val = 0

    # 1. Snake Matrix Score
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            if val > 0:
                snake_score += val * SNAKE_MATRIX[r, c]
                if val > max_val:
                    max_val = val
            else:
                empty_count += 1

    # 2. Smoothness Penalty (Kara za sąsiadujące kafelki różniące się wartością)
    for r in range(4):
        for c in range(4):
            val = board[r, c]
            if val > 0:
                log_val = np.log2(val)
                if c + 1 < 4 and board[r, c + 1] > 0:
                    smoothness -= abs(log_val - np.log2(board[r, c + 1]))
                if r + 1 < 4 and board[r + 1, c] > 0:
                    smoothness -= abs(log_val - np.log2(board[r + 1, c]))

    # 3. Potęgowe skalowanie pustych pól
    empty_score = (empty_count ** 2) * (max_val if max_val > 0 else 1) * 10.0

    return snake_score + empty_score + (smoothness * 500.0)

@njit(float64(uint64), fastmath=True)
def evaluate_d4_v7(board64):
    board = bitboard_to_numpy(board64)
    max_eval = -1e18
    b_curr = board.copy()

    for rot in range(4):
        e1 = evaluate_sota_v7_aspect(b_curr)
        if e1 > max_eval:
            max_eval = e1
        b_flip = np.fliplr(b_curr)
        e2 = evaluate_sota_v7_aspect(b_flip)
        if e2 > max_eval:
            max_eval = e2
        b_curr = np.rot90(b_curr, 1)

    return max_eval

# === 5. EXPECTIMAX V7 Z PRZEKAZYWANIEM CACHE ===

@njit(float64(uint64, int64, boolean, uint64[:], int8[:], float64[:]), fastmath=True)
def expectimax_v7_core(board, depth, is_player, h_board, h_depth, h_score):
    if depth == 0:
        return evaluate_d4_v7(board)

    hash_idx = int(board % np.uint64(TABLE_SIZE))
    if h_board[hash_idx] == board and h_depth[hash_idx] >= depth:
        return h_score[hash_idx]

    if is_player:
        best_score = -1e18
        moved = False
        for a in range(4):
            b_next, _, changed = execute_move_bitboard(board, a)
            if changed:
                moved = True
                score = expectimax_v7_core(b_next, depth - 1, False, h_board, h_depth, h_score)
                if score > best_score:
                    best_score = score

        res_score = best_score if moved else -1e12

        h_board[hash_idx] = board
        h_depth[hash_idx] = np.int8(depth)
        h_score[hash_idx] = res_score
        return res_score
    else:
        empty_positions = np.zeros(16, dtype=np.int64)
        n_empty = 0

        for i in range(16):
            if ((board >> (i * 4)) & np.uint64(0x0F)) == 0:
                empty_positions[n_empty] = i
                n_empty += 1

        if n_empty == 0:
            return evaluate_d4_v7(board)

        if n_empty > 3 and depth >= 3:
            n_empty = 3

        expected = 0.0
        for idx in range(n_empty):
            pos = empty_positions[idx]
            shift = pos * 4
            b2 = board | (np.uint64(1) << shift)
            b4 = board | (np.uint64(2) << shift)

            res = 0.9 * expectimax_v7_core(b2, depth - 1, True, h_board, h_depth, h_score) + 0.1 * expectimax_v7_core(b4, depth - 1, True, h_board, h_depth, h_score)
            expected += res

        final_val = expected / float(n_empty)
        return final_val

def numpy_to_bitboard(board_np):
    b64 = np.uint64(0)
    for r in range(4):
        for c in range(4):
            val = board_np[r, c]
            if val > 0:
                power = int(np.log2(val))
                shift = (r * 4 + c) * 4
                b64 |= (np.uint64(power) << shift)
    return b64

def get_v7_god_move(board_np, valid_mask=None):
    b64 = numpy_to_bitboard(board_np)
    empty_tiles = np.sum(board_np == 0)

    # LATE GAME GŁĘBOKOŚĆ DEPTH 7 I 8!
    if empty_tiles <= 2:
        target_depth = 7  # Ekstremalny podgląd 7 kroków!
    elif empty_tiles <= 4:
        target_depth = 6  # Głęboka analiza
    else:
        target_depth = 5  # Standardowa głębokość

    best_action = -1
    best_score = -1e18

    for a in range(4):
        if valid_mask is not None and not valid_mask[a]:
            continue

        b_next, _, changed = execute_move_bitboard(b64, a)
        if changed:
            score = expectimax_v7_core(b_next, target_depth - 1, False, HASH_BOARD, HASH_DEPTH, HASH_SCORE)
            if score > best_score:
                best_score = score
                best_action = a

    if best_action == -1:
        if valid_mask is not None and np.any(valid_mask):
            best_action = int(np.where(valid_mask)[0][0])
        else:
            best_action = 0

    return best_action, target_depth
