import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from game2048_v5_bitenv_fast import Game2048BitboardEnvFast
from bitboard_v5_god import execute_move_bitboard, numpy_to_bitboard

# === 1. SYNERGIA ALPHAZERO: EVALUACJA ŁĄCZĄCA PYTORCH NEURAL VALUE + BITBOARD O(1) ===

class AlphaZeroV5Engine:
    def __init__(self, model_path="models/maskable_ppo_v5_10m"):
        print(f"🧠 Ładowanie wytrenowanej sieci neuronowej Value Network: {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = MaskablePPO.load(model_path, device=self.device)
            print(f"✅ Sieć neuronowa gotowa na {self.device.upper()}!")
        except Exception:
            print("⚠️ Nie znaleziono zapisanej sieci V5, używam domyślnej ewaluacji hybrydowej.")
            self.model = None

    def evaluate_board_neural(self, board_np):
        """Ocena stanu planszy przez sieć neuronową Value Network PPO V(s)."""
        if self.model is None:
            return float(np.sum(board_np) + np.sum(board_np == 0) * 100.0)

        # Generowanie obserwacji One-Hot dla sieci
        obs = np.zeros((1, 16, 4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                val = board_np[r, c]
                if val > 0:
                    power = int(np.log2(val))
                    if power < 16:
                        obs[0, power, r, c] = 1.0
                    else:
                        obs[0, 15, r, c] = 1.0
                else:
                    obs[0, 0, r, c] = 1.0

        obs_tensor = torch.as_tensor(obs).to(self.device)
        with torch.no_grad():
            values = self.model.policy.predict_values(obs_tensor)
            val_score = values.cpu().numpy()[0][0]

        return float(val_score)

    def expectimax_alphazero(self, board_np, depth, is_player):
        if depth == 0:
            return self.evaluate_board_neural(board_np)

        if is_player:
            best_score = -1e18
            b64 = numpy_to_bitboard(board_np)
            moved = False

            for a in range(4):
                b_next_64, _, changed = execute_move_bitboard(b64, a)
                if changed:
                    moved = True
                    # Unpack do numpy dla ewaluacji
                    next_np = np.zeros((4, 4), dtype=np.int32)
                    for r in range(4):
                        for c in range(4):
                            shift = (r * 4 + c) * 4
                            pow_v = int((b_next_64 >> shift) & np.uint64(0x0F))
                            if pow_v > 0:
                                next_np[r, c] = 1 << pow_v

                    score = self.expectimax_alphazero(next_np, depth - 1, False)
                    if score > best_score:
                        best_score = score
            return best_score if moved else -10000.0
        else:
            empty_positions = list(zip(*np.where(board_np == 0)))
            if not empty_positions:
                return self.evaluate_board_neural(board_np)

            if len(empty_positions) > 3 and depth >= 2:
                import random
                empty_positions = random.sample(empty_positions, 3)

            expected = 0.0
            for pos in empty_positions:
                b2 = board_np.copy()
                b2[pos] = 2
                expected += 0.9 * self.expectimax_alphazero(b2, depth - 1, True)

                b4 = board_np.copy()
                b4[pos] = 4
                expected += 0.1 * self.expectimax_alphazero(b4, depth - 1, True)

            return expected / len(empty_positions)

    def get_move(self, board_np, depth=3):
        best_action = 0
        best_score = -1e18
        b64 = numpy_to_bitboard(board_np)

        for a in range(4):
            b_next_64, _, changed = execute_move_bitboard(b64, a)
            if changed:
                next_np = np.zeros((4, 4), dtype=np.int32)
                for r in range(4):
                    for c in range(4):
                        shift = (r * 4 + c) * 4
                        pow_v = int((b_next_64 >> shift) & np.uint64(0x0F))
                        if pow_v > 0:
                            next_np[r, c] = 1 << pow_v

                score = self.expectimax_alphazero(next_np, depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_action = a

        return best_action

def run_alphazero_demo(n_games=3, depth=3):
    engine = AlphaZeroV5Engine()
    print(f"\n🚀 ROZPOZCZYNAM TURNIEJ ALPHAZERO (Neural Value Network + Bitboard O(1))...\n")

    scores = []
    max_tiles = []

    for g in range(n_games):
        env = Game2048BitboardEnvFast()
        obs, info = env.reset()
        done = False
        start_time = time.time()

        while not done:
            action = engine.get_move(env.board, depth=depth)
            obs, reward, done, truncated, info = env.step(action)

        elapsed = time.time() - start_time
        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        print(f"  🏆 Gra #{g+1}: Wynik = {info['score']:>8,} | Max Kafelek = {info['max_tile']:>5} | Czas = {elapsed:.1f}s | Ruchy = {info['move_count']}")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print(f"\n{'='*60}")
    print(f"  👑 WYNIKI ALPHAZERO HYBRID V5")
    print(f"{'='*60}")
    print(f"  Średni wynik:         {scores.mean():>10,.0f}")
    print(f"  Rekord punktowy:      {scores.max():>10,.0f}")
    print(f"  Średni max kafelek:   {max_tiles.mean():>10,.0f}")
    print(f"  Najwyższy kafelek:    {max_tiles.max():>10,.0f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_alphazero_demo(n_games=3, depth=3)
