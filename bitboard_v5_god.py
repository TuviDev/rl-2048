import time
import numpy as np
from numba import njit, uint16, uint32, uint64, int64, float64, boolean, types

# === PREKOMPILACJA TABLIC LOOKUP O(1) NA TYPACH UINT64 ===

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

        # --- LEWO ---
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

        # --- PRAWO (POPRAWIONE ODBICIE RUCHU) ---
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

# === OPERACJE BITOWE O(1) ===

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

# === HEURYSTYKA MATRYCY WĘŻA SOTA ===

SNAKE_WEIGHTS = np.array([
    [1073741824.0, 268435456.0, 67108864.0, 16777216.0],
    [65536.0,      262144.0,    1048576.0,  4194304.0],
    [32768.0,      8192.0,      2048.0,     512.0],
    [16.0,         32.0,        64.0,       128.0]
], dtype=np.float64)

@njit(float64(uint64), fastmath=True)
def evaluate_bitboard_sota(board):
    score = 0.0
    empty_count = 0
    max_power = 0
    max_pos = 0

    for i in range(16):
        shift = i * 4
        power = int((board >> shift) & np.uint64(0x0F))
        if power == 0:
            empty_count += 1
        else:
            val = 1 << power
            r = i // 4
            c = i % 4
            score += val * SNAKE_WEIGHTS[r, c]
            if power > max_power:
                max_power = power
                max_pos = i

    score += empty_count * 500000.0

    if max_pos == 0:
        score += 10000000.0
    else:
        score -= 10000000.0

    return score

@njit(float64(uint64, int64, boolean), fastmath=True)
def expectimax_v5(board, depth, is_player):
    if depth == 0:
        return evaluate_bitboard_sota(board)

    if is_player:
        best_score = -1e18
        moved = False
        for a in range(4):
            b_next, _, changed = execute_move_bitboard(board, a)
            if changed:
                moved = True
                score = expectimax_v5(b_next, depth - 1, False)
                if score > best_score:
                    best_score = score
        return best_score if moved else -1e12
    else:
        empty_positions = np.zeros(16, dtype=np.int64)
        n_empty = 0

        for i in range(16):
            if ((board >> (i * 4)) & np.uint64(0x0F)) == 0:
                empty_positions[n_empty] = i
                n_empty += 1

        if n_empty == 0:
            return evaluate_bitboard_sota(board)

        expected = 0.0
        for idx in range(n_empty):
            pos = empty_positions[idx]
            shift = pos * 4
            b2 = board | (np.uint64(1) << shift)
            b4 = board | (np.uint64(2) << shift)

            res = 0.9 * expectimax_v5(b2, depth - 1, True) + 0.1 * expectimax_v5(b4, depth - 1, True)
            expected += res

        return expected / float(n_empty)

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

def get_v5_ultimate_move(board_np, valid_mask=None):
    b64 = numpy_to_bitboard(board_np)
    empty_tiles = np.sum(board_np == 0)

    if empty_tiles <= 4:
        target_depth = 6
    else:
        target_depth = 5

    best_action = -1
    best_score = -1e18

    for a in range(4):
        if valid_mask is not None and not valid_mask[a]:
            continue

        b_next, _, changed = execute_move_bitboard(b64, a)
        if changed:
            score = expectimax_v5(b_next, target_depth - 1, False)
            if score > best_score:
                best_score = score
                best_action = a

    if best_action == -1:
        if valid_mask is not None and np.any(valid_mask):
            best_action = int(np.where(valid_mask)[0][0])
        else:
            best_action = 0

    return best_action, target_depth
