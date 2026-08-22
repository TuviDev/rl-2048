import numpy as np

print("\n" + "="*60)
print("🔬 DIAGNOSTYKA I TESTY JEDNOSTKOWE SILNIKÓW V4 VS V5")
print("="*60 + "\n")

# 1. Testowa plansza
test_board = np.array([
    [2, 2, 0, 0],
    [4, 4, 4, 4],
    [0, 0, 8, 8],
    [2, 4, 8, 16]
], dtype=np.int64)

print("📋 Testowa plansza wejściowa:")
print(test_board)
print("-" * 40)

# Import V4 Engine (NumPy)
try:
    from expectimax_god_v4 import simulate_move_numba as move_v4
    print("✅ Silnik V4 (NumPy Numba): Załadowany pomyślnie")
except Exception as e:
    print(f"❌ Błąd V4: {e}")

# Import V5 Engine (Bitboard)
try:
    from bitboard_v5_god import execute_move_bitboard, numpy_to_bitboard
    print("✅ Silnik V5 (Bitboard): Załadowany pomyślnie")
except Exception as e:
    print(f"❌ Błąd V5: {e}")

print("\n🧪 Porównanie wyników symulacji ruchu w PRAWO (Action 2):")

# Test V4
b4_res, score4, ch4 = move_v4(test_board, 2)
print("\nOutput V4 (Poprawny):")
print(b4_res)
print(f"Score V4: {score4} | Changed: {ch4}")

# Test V5
b5_in = numpy_to_bitboard(test_board)
b5_res_bit, score5, ch5 = execute_move_bitboard(b5_in, 2)

# Unpack bitboard do macierzy
def unpack_test(b64):
    res = np.zeros((4, 4), dtype=np.int64)
    for r in range(4):
        for c in range(4):
            shift = (r * 4 + c) * 4
            power = int((b64 >> shift) & np.uint64(0x0F))
            if power > 0:
                res[r, c] = 1 << power
    return res

b5_res = unpack_test(b5_res_bit)
print("\nOutput V5 (Bitboard):")
print(b5_res)
print(f"Score V5: {score5} | Changed: {ch5}")

print("\n" + "="*60)
if np.array_equal(b4_res, b5_res):
    print("🎉 WYNIK TESTU: Silniki V4 i V5 są w 100% ZGODNE!")
else:
    print("⚠️ WYKRYTO ROZBIEŻNOŚĆ! Silnik V5 deformuje planszę!")
print("="*60 + "\n")
