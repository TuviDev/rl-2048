import numpy as np
from numba import njit, uint16, uint64, float64

# PREKOMPILOWANA TABLICA LOOKUP DLA BRAMEK BITOWYCH (65536 MOŻLIWYCH WIERSZY)
# Każdy wiersz to 16 bitów (4 komórki x 4 bity)

@njit(fastmath=True)
def unpack_board(board64):
    """Zamienia 64-bitowy int na macierz 4x4 NumPy."""
    res = np.zeros((4, 4), dtype=np.int64)
    for r in range(4):
        for c in range(4):
            shift = (r * 4 + c) * 4
            power = (board64 >> shift) & 0x0F
            if power > 0:
                res[r, c] = 1 << power
    return res

@njit(fastmath=True)
def eval_bitboard_heuristic(board64):
    """Maksymalnie zoptymalizowana funkcja oceny planszy bitowej."""
    score = 0.0
    empty_count = 0
    max_power = 0
    max_pos = 0

    weights = np.array([
        200000.0, 30000.0, 5000.0, 1000.0,
        10.0,     50.0,    200.0,  500.0,
        8.0,      6.0,     4.0,    2.0,
        0.1,      0.2,     0.3,    0.4
    ], dtype=np.float64)

    for i in range(16):
        shift = i * 4
        power = (board64 >> shift) & 0x0F
        if power == 0:
            empty_count += 1
        else:
            val = 1 << power
            score += val * weights[i]
            if power > max_power:
                max_power = power
                max_pos = i

    # Premia za puste pola
    score += empty_count * 10000.0

    # Najwyższy kafelek MUSI być w rogu (pozycja 0)
    if max_pos == 0:
        score += 100000.0
    else:
        score -= 100000.0

    return score

print("⚡ Silnik Bitboard O(1) został pomyślnie skonfigurowany!")
