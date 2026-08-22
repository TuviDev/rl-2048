import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Game2048Env(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.size = 4
        self.observation_space = spaces.Box(
            low=0, high=15, shape=(self.size, self.size), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        self.action_names = {0: "← LEWO", 1: "↑ GÓRA", 2: "→ PRAWO", 3: "↓ DÓŁ"}
        self.board = None
        self.score = 0
        self.max_tile = 0
        self.move_count = 0
        self.consecutive_invalid_moves = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.move_count = 0
        self.consecutive_invalid_moves = 0
        self._add_random_tile()
        self._add_random_tile()
        return self._get_obs(), self._get_info()

    def step(self, action):
        old_board = self.board.copy()
        merge_reward = self._move(action)
        board_changed = not np.array_equal(old_board, self.board)

        if board_changed:
            self._add_random_tile()
            self.move_count += 1
            self.consecutive_invalid_moves = 0
        else:
            self.consecutive_invalid_moves += 1

        current_max = self.board.max() if self.board.max() > 0 else 0
        self.max_tile = max(self.max_tile, current_max)

        # Koniec gry: brak ruchów LUB 4 zacięcia z rzędu (brak możliwości ruchu w żadną stronę)
        done = self._is_game_over() or (self.consecutive_invalid_moves >= 4)

        reward = float(merge_reward)
        reward += float(np.sum(self.board == 0)) * 1.0
        reward += self._monotonicity_score() * 0.1

        if not board_changed:
            reward -= 5.0
        if done and self._is_game_over():
            reward -= 50.0

        return self._get_obs(), reward, done, False, self._get_info()

    def _get_obs(self):
        obs = np.zeros_like(self.board, dtype=np.int32)
        mask = self.board > 0
        obs[mask] = np.log2(self.board[mask]).astype(np.int32)
        return obs

    def _get_info(self):
        return {
            "score": int(self.score),
            "max_tile": int(self.max_tile),
            "move_count": int(self.move_count),
            "empty_cells": int(np.sum(self.board == 0)),
        }

    def _add_random_tile(self):
        empty_positions = list(zip(*np.where(self.board == 0)))
        if not empty_positions:
            return
        pos = random.choice(empty_positions)
        self.board[pos] = 2 if random.random() < 0.9 else 4

    def _move(self, direction):
        merge_reward = 0
        if direction == 0:
            rotated = self.board.copy()
        elif direction == 1:
            rotated = np.rot90(self.board, k=-1)
        elif direction == 2:
            rotated = np.rot90(self.board, k=2)
        elif direction == 3:
            rotated = np.rot90(self.board, k=1)

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
                    self.score += new_val
                    skip = True
                else:
                    merged.append(non_zero[j])
            new_board[i, :len(merged)] = merged

        if direction == 0:
            self.board = new_board
        elif direction == 1:
            self.board = np.rot90(new_board, k=1)
        elif direction == 2:
            self.board = np.rot90(new_board, k=2)
        elif direction == 3:
            self.board = np.rot90(new_board, k=-1)

        return merge_reward

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

    def _monotonicity_score(self):
        score = 0.0
        log_board = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        log_board[mask] = np.log2(self.board[mask])

        for axis in [0, 1]:
            for direction in [1, -1]:
                s = 0.0
                for i in range(self.size):
                    if axis == 0:
                        row = log_board[i, ::direction]
                    else:
                        row = log_board[::direction, i]
                    for j in range(self.size - 1):
                        if row[j] >= row[j + 1]:
                            s += row[j] - row[j + 1]
                score = max(score, s)
        return score

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

    def close(self):
        pass
