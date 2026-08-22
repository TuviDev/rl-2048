import time
import numpy as np
import torch
from stable_baselines3 import PPO
from game2048_v2_env import Game2048V2Env

class HybridMasterAI:
    def __init__(self, model_path="models/ppo_2048_cnn_v2", depth=3):
        print(f"🧠 Ładowanie sieci neuronowej: {model_path}...")
        self.model = PPO.load(model_path)
        self.depth = depth

    def evaluate_board(self, board):
        env = Game2048V2Env()
        env.board = board.copy()
        obs = env._get_obs()
        obs_tensor = torch.as_tensor(obs[None]).to(self.model.device)
        
        with torch.no_grad():
            value = self.model.policy.predict_values(obs_tensor).cpu().numpy()[0][0]

        empty = np.sum(board == 0)
        max_val = np.max(board)
        is_corner = board[3, 0] == max_val or board[3, 3] == max_val or board[0, 0] == max_val or board[0, 3] == max_val
        
        score = float(value) + (empty * 15.0) + (50.0 if is_corner else -30.0)
        return score

    def expectimax(self, board, depth, is_player):
        if depth == 0:
            return self.evaluate_board(board)

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
                    score = self.expectimax(temp_env.board, depth - 1, is_player=False)
                    best_score = max(best_score, score)
            return best_score if moved else -10000.0
        else:
            empty_positions = list(zip(*np.where(board == 0)))
            if not empty_positions:
                return self.evaluate_board(board)

            if len(empty_positions) > 3:
                import random
                empty_positions = random.sample(empty_positions, 3)

            expected_score = 0.0
            for pos in empty_positions:
                b2 = board.copy()
                b2[pos] = 2
                expected_score += 0.9 * self.expectimax(b2, depth - 1, is_player=True)

                b4 = board.copy()
                b4[pos] = 4
                expected_score += 0.1 * self.expectimax(b4, depth - 1, is_player=True)

            return expected_score / len(empty_positions)

    def get_best_move(self, board):
        best_action = 0
        best_score = -float('inf')

        for action in range(4):
            temp_env = Game2048V2Env()
            temp_env.board = board.copy()
            old_b = temp_env.board.copy()
            temp_env._move(action)
            if not np.array_equal(old_b, temp_env.board):
                score = self.expectimax(temp_env.board, self.depth - 1, is_player=False)
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

def evaluate_master(n_games=5, depth=3):
    ai = HybridMasterAI(depth=depth)
    scores = []
    max_tiles = []

    print(f"\n🚀 TURNIEJ POTĘŻNEJ HYBRYDY (Głębokość analizy: {depth} ruchy w przód)...\n")

    for g in range(n_games):
        env = Game2048V2Env()
        obs, info = env.reset()
        done = False

        while not done:
            action = ai.get_best_move(env.board)
            obs, reward, done, truncated, info = env.step(action)

        scores.append(info["score"])
        max_tiles.append(info["max_tile"])
        print(f"  Gra #{g+1:>2}: Score: {info['score']:>7,} | Max Kafelek: {info['max_tile']:>5} | Ruchy: {info['move_count']:>4}")

    scores = np.array(scores)
    max_tiles = np.array(max_tiles)

    print(f"\n{'='*55}")
    print(f"  🏆 WYNIKI HYBRYDY DEPTH={depth}")
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

def watch_live(depth=3, delay=0.05):
    ai = HybridMasterAI(depth=depth)
    env = Game2048V2Env()
    obs, info = env.reset()
    done = False
    step = 0

    print(f"🎬 OGLĄDAMY MECZ NA ŻYWO (Depth={depth})!\n")
    env.render()
    time.sleep(1)

    while not done:
        action = ai.get_best_move(env.board)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        print(f"Ruch #{step}: {env.action_names[action]}")
        env.render()
        time.sleep(delay)

    print(f"\n🏁 KONIEC ROZGRYWKI!")
    print(f"Finalny Score: {info['score']:,} | Max kafelek: {info['max_tile']}")

if __name__ == "__main__":
    evaluate_master(n_games=3, depth=3)
    input("\nNaciśnij [ENTER], aby obejrzeć mecz na żywo z Depth=3...")
    watch_live(depth=3)
