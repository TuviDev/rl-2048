import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Game2048V3MaskEnv(gym.Env):
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
        
        self.board = None
        self.score = 0
        self.max_tile = 0
        self.move_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.move_count = 0
        self._add_random_tile()
        self._add_random_tile()
        return self._get_obs(), self._get_info()

    def action_masks(self) -> np.ndarray:
        masks = np.zeros(4, dtype=bool)
        for a in range(4):
            temp_b = self.board.copy()
            merged_score, changed = self._simulate_move(temp_b, a)
            if changed:
                masks[a] = True
        if not np.any(masks):
            return np.ones(4, dtype=bool)
        return masks

    def step(self, action):
        action = int(action)
        merge_score, board_changed = self._simulate_move(self.board, action)

        if board_changed:
            self.score += merge_score  # <-- TUTAJ NAPRAWIONO ZLICZANIE PUKNTÓW!
            self._add_random_tile()
            self.move_count += 1

        current_max = int(self.board.max()) if self.board.max() > 0 else 0
        self.max_tile = max(self.max_tile, current_max)

        done = self._is_game_over() or not np.any(self.action_masks())

        reward = 0.0
        if merge_score > 0:
            reward += np.log2(merge_score) * 2.0

        empty_count = np.sum(self.board == 0)
        reward += empty_count * 0.3

        if done and self._is_game_over():
            reward -= 20.0

        return self._get_obs(), float(reward), done, False, self._get_info()

    def _get_obs(self):
        obs = np.zeros((16, self.size, self.size), dtype=np.float32)
        for r in range(self.size):
            for c in range(self.size):
                val = self.board[r, c]
                if val == 0:
                    obs[0, r, c] = 1.0
                else:
                    power = int(np.log2(val))
                    if power < 16:
                        obs[power, r, c] = 1.0
                    else:
                        obs[15, r, c] = 1.0
        return obs

    def _get_info(self):
        return {
            "score": int(self.score),
            "max_tile": int(self.max_tile),
            "move_count": int(self.move_count),
        }

    def _add_random_tile(self):
        empty_positions = list(zip(*np.where(self.board == 0)))
        if not empty_positions:
            return
        pos = random.choice(empty_positions)
        self.board[pos] = 2 if random.random() < 0.9 else 4

    def _simulate_move(self, board_ref, direction):
        merge_reward = 0
        if direction == 0:
            rotated = board_ref.copy()
        elif direction == 1:
            rotated = np.rot90(board_ref, k=-1)
        elif direction == 2:
            rotated = np.rot90(board_ref, k=2)
        elif direction == 3:
            rotated = np.rot90(board_ref, k=1)

        new_board = np.zeros_like(rotated)
        for i in range(self.size):
            row = rotated[i]
            non_zero = row[row != 0]
            merged = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    new_val = non_zero[j] * 2
                    merged.append(new_val)
                    merge_reward += new_val
                    skip = True
                else:
                    merged.append(non_zero[j])
            new_board[i, :len(merged)] = merged

        if direction == 0:
            unrotated = new_board
        elif direction == 1:
            unrotated = np.rot90(new_board, k=1)
        elif direction == 2:
            unrotated = np.rot90(new_board, k=2)
        elif direction == 3:
            unrotated = np.rot90(new_board, k=-1)

        changed = not np.array_equal(board_ref, unrotated)
        if changed:
            board_ref[:] = unrotated

        return merge_reward, changed

    def _is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i, j]
                if j + 1 < self.size and self.board[i, j + 1] == val:
                    return False
                if i + 1 < self.size and self.board[i + 1, j] == val:
                    return False
        return True

    def render(self):
        print(f"\n{'='*29}")
        print(f"  Score: {self.score:>6}  |  Max: {self.max_tile:>5}")
        print(f"{'='*29}")
        for i in range(self.size):
            row_str = "|"
            for j in range(self.size):
                val = self.board[i, j]
                if val == 0:
                    row_str += "    .|"
                else:
                    row_str += f"{val:>5}|"
            print(row_str)
        print(f"{'='*29}\n")
