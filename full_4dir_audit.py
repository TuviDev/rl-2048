import numpy as np

print("\n" + "="*60)
print("🔬 PEŁNY AUDYT DIAGNOSTYCZNY 4 KIERUNKÓW (V4 VS V5)")
print("="*60 + "\n")

from expectimax_god_v4 import simulate_move_numba as move_v4
from bitboard_v5_god import execute_move_bitboard, numpy_to_bitboard, bitboard_to_numpy

# 5 Trudnych plansz testowych
test_boards = [
    np.array([[2, 2, 4, 8], [4, 4, 8, 16], [8, 8, 16, 32], [16, 32, 64, 128]], dtype=np.int64),
    np.array([[0, 2, 2, 4], [0, 0, 4, 8], [2, 4, 8, 16], [4, 8, 16, 32]], dtype=np.int64),
    np.array([[2, 0, 0, 2], [4, 0, 0, 4], [8, 0, 0, 8], [16, 0, 0, 16]], dtype=np.int64),
    np.array([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [2, 0, 0, 0]], dtype=np.int64),
    np.array([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [16, 8, 4, 2]], dtype=np.int64)
]

action_names = ["0: LEWO", "1: GÓRA", "2: PRAWO", "3: DÓŁ"]
total_mismatches = 0

for b_idx, board in enumerate(test_boards):
    print(f"📋 PLANSZA TESTOWA #{b_idx + 1}:")
    print(board)
    
    b5_in = numpy_to_bitboard(board)
    
    for a in range(4):
        # Symulacja V4
        res_v4, score_v4, changed_v4 = move_v4(board, a)
        
        # Symulacja V5
        res_v5_bit, score_v5, changed_v5 = execute_move_bitboard(b5_in, a)
        res_v5 = bitboard_to_numpy(res_v5_bit)
        
        # Porównanie
        match_board = np.array_equal(res_v4, res_v5)
        match_score = (score_v4 == score_v5)
        match_changed = (changed_v4 == changed_v5)
        
        if not (match_board and match_score and match_changed):
            total_mismatches += 1
            print(f"  ❌ ROZBIEŻNOŚĆ dla akcji [{action_names[a]}]:")
            print(f"     V4 Result:\n{res_v4}")
            print(f"     V5 Result:\n{res_v5}")
            print(f"     Score V4: {score_v4} vs V5: {score_v5}")
            print(f"     Changed V4: {changed_v4} vs V5: {changed_v5}")
        else:
            print(f"  ✅ Akcja [{action_names[a]}]: ZGODNA")
    print("-" * 50)

print("\n" + "="*60)
if total_mismatches == 0:
    print("🎉 WYNIK AUDYTU: Wszystkie 20 testów kierunkowych przeszły idealnie!")
else:
    print(f"⚠️ Wykryto {total_mismatches} błędów w symulacji ruchów V5!")
print("="*60 + "\n")
