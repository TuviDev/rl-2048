import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from numba import njit, uint64, uint32, int64, float32, boolean, types

# === PREKOMPILOWANE TABLICE BITBOARD O(1) ===
def _build_bitboard_tables():
    row_left = np.zeros(65536, dtype=np.uint64)
    row_score = np.zeros(65536, dtype=np.uint32)

    for row_val in range(65536):
        cells = [
            (row_val >> 0) & 0x0F,
            (row_val >> 4) & 0x0F,
            (row_val >> 8) & 0x0F,
            (row_val >> 12) & 0x0F
        ]

        non_zero = [c for c in cells if c > 0]
        merged = []
        score = 0
        skip = False

        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                npow = non_zero[i] + 1
                merged.append(npow)
                score += (1 << npow)
                skip = True
            else:
                merged.append(non_zero[i])

        while len(merged) < 4:
            merged.append(0)

        left_v = np.uint64(merged[0] | (merged[1] << 4) | (merged[2] << 8) | (merged[3] << 12))
        row_left[row_val] = left_v
        row_score[row_val] = np.uint32(score)

    return row_left, row_score

ROW_LEFT_TABLE, ROW_SCORE_TABLE = _build_bitboard_tables()

@njit(uint64(uint64), fastmath=True)
def _transpose_bits(board):
    r0 = (board >> 0) & 0xFFFF
    r1 = (board >> 16) & 0xFFFF
    r2 = (board >> 32) & 0xFFFF
    r3 = (board >> 48) & 0xFFFF

    c0 = (r0 & 0x0F) | ((r1 & 0x0F) << 4) | ((r2 & 0x0F) << 8) | ((r3 & 0x0F) << 12)
    c1 = ((r0 >> 4) & 0x0F) | (((r1 >> 4) & 0x0F) << 4) | (((r2 >> 4) & 0x0F) << 8) | (((r3 >> 4) & 0x0F) << 12)
    c2 = ((r0 >> 8) & 0x0F) | (((r1 >> 8) & 0x0F) << 4) | (((r2 >> 8) & 0x0F) << 8) | (((r3 >> 8) & 0x0F) << 12)
    c3 = ((r0 >> 12) & 0x0F) | (((r1 >> 12) & 0x0F) << 4) | (((r2 >> 12) & 0x0F) << 8) | (((r3 >> 12) & 0x0F) << 12)

    return np.uint64(c0) | (np.uint64(c1) << 16) | (np.uint64(c2) << 32) | (np.uint64(c3) << 48)

@njit(uint64(uint64), fastmath=True)
def _flip_h_bits(board):
    n0 = (board & np.uint64(0x000F000F000F000F)) << 12
    n1 = (board & np.uint64(0x00F000F000F000F0)) << 4
    n2 = (board & np.uint64(0x0F000F000F000F00)) >> 4
    n3 = (board & np.uint64(0xF000F000F000F000)) >> 12
    return n0 | n1 | n2 | n3

@njit(types.Tuple((uint64, uint32, boolean))(uint64, int64), fastmath=True)
def bitboard_step_fast(board, action):
    b = board
    if action == 1:
        b = _transpose_bits(b)
    elif action == 2:
        b = _flip_h_bits(b)
    elif action == 3:
        b = _flip_h_bits(_transpose_bits(b))

    r0 = int((b >> 0) & 0xFFFF)
    r1 = int((b >> 16) & 0xFFFF)
    r2 = int((b >> 32) & 0xFFFF)
    r3 = int((b >> 48) & 0xFFFF)

    nr0 = ROW_LEFT_TABLE[r0]
    nr1 = ROW_LEFT_TABLE[r1]
    nr2 = ROW_LEFT_TABLE[r2]
    nr3 = ROW_LEFT_TABLE[r3]

    new_b = np.uint64(nr0) | (np.uint64(nr1) << 16) | (np.uint64(nr2) << 32) | (np.uint64(nr3) << 48)

    if action == 1:
        new_b = _transpose_bits(new_b)
    elif action == 2:
        new_b = _flip_h_bits(new_b)
    elif action == 3:
        new_b = _transpose_bits(_flip_h_bits(new_b))

    score = ROW_SCORE_TABLE[r0] + ROW_SCORE_TABLE[r1] + ROW_SCORE_TABLE[r2] + ROW_SCORE_TABLE[r3]
    changed = (new_b != board)

    return new_b, score, changed

# BŁYSKAWICZNE TWORZENIE OBSERWACJI ONE-HOT W NUMBA C++
@njit(float32[:, :, :](uint64), fastmath=True)
def get_obs_numba(board64):
    obs = np.zeros((16, 4, 4), dtype=np.float32)
    for i in range(16):
        r = i // 4
        c = i % 4
        power = int((board64 >> (i * 4)) & np.uint64(0x0F))
        if power < 16:
            obs[power, r, c] = 1.0
        else:
            obs[15, r, c] = 1.0
    return obs


class Game2048BitboardEnvFast(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.size = 4

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16, self.size, self.size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.action_names = {0: "← LEWO", 1: "↑ GÓRA", 2: "→ PRAWO", 3: "↓ DÓŁ"}
        
        self.board64 = np.uint64(0)
        self.board_np = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.move_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board64 = np.uint64(0)
        self.board_np = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.move_count = 0
        
        self._add_random_tile_bitboard()
        self._add_random_tile_bitboard()
        return get_obs_numba(self.board64), self._get_info()

    def action_masks(self) -> np.ndarray:
        masks = np.zeros(4, dtype=bool)
        for a in range(4):
            _, _, changed = bitboard_step_fast(self.board64, a)
            if changed:
                masks[a] = True
        if not np.any(masks):
            return np.ones(4, dtype=bool)
        return masks

    def step(self, action):
        action = int(action)
        new_b64, merge_score, changed = bitboard_step_fast(self.board64, action)

        if changed:
            self.board64 = new_b64
            self.score += int(merge_score)
            self.move_count += 1
            self._add_random_tile_bitboard()

        done = not np.any(self.action_masks())

        reward = 0.0
        if merge_score > 0:
            reward += np.log2(merge_score) * 2.0

        if done:
            reward -= 10.0

        return get_obs_numba(self.board64), float(reward), done, False, self._get_info()

    def _get_info(self):
        return {
            "score": int(self.score),
            "max_tile": int(self.max_tile),
            "move_count": int(self.move_count),
        }

    def _add_random_tile_bitboard(self):
        empty_positions = []
        for i in range(16):
            if ((self.board64 >> (i * 4)) & np.uint64(0x0F)) == 0:
                empty_positions.append(i)

        if not empty_positions:
            return

        pos = random.choice(empty_positions)
        shift = pos * 4
        val_power = 1 if random.random() < 0.9 else 2
        self.board64 |= (np.uint64(val_power) << shift)

    @property
    def board(self):
        self.board_np.fill(0)
        for r in range(4):
            for c in range(4):
                shift = (r * 4 + c) * 4
                power = int((self.board64 >> shift) & np.uint64(0x0F))
                if power > 0:
                    self.board_np[r, c] = 1 << power
        return self.board_np
