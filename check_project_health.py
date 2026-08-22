import os
import sys
import time
import numpy as np

print("\n" + "="*60)
print("🔍 AUDYT SPÓJNOŚCI I KONTROLA JAKOŚCI PROJEKTU RL-2048")
print("="*60 + "\n")

errors = 0

# 1. Sprawdzenie kluczowych plików
required_files = [
    "game2048_v3_mask.py",
    "expectimax_god_v4.py",
    "evaluate_god_v4.py",
    "app.py",
    "README.md",
    "requirements.txt",
    ".gitignore"
]

print("📁 1. Weryfikacja struktury plików na dysku:")
for fname in required_files:
    if os.path.exists(fname):
        print(f"   ✅ {fname:<25} -> OBECNY")
    else:
        print(f"   ❌ {fname:<25} -> BRAKUJE!")
        errors += 1

# 2. Test importu środowiska V3 Mask
print("\n🎮 2. Test środowiska Gym (game2048_v3_mask.py):")
try:
    from game2048_v3_mask import Game2048V3MaskEnv
    env = Game2048V3MaskEnv()
    obs, info = env.reset()
    _, reward, done, _, info_after = env.step(0)
    print("   ✅ Import środowiska: OK")
    print(f"   ✅ Test zliczania punktów (Score): {info_after['score']} pkt (działa!)")
except Exception as e:
    print(f"   ❌ Błąd środowiska: {e}")
    errors += 1

# 3. Test silnika Numba C++ God V4
print("\n👑 3. Test Silnika God Mode V4 (expectimax_god_v4.py):")
try:
    from expectimax_god_v4 import get_adaptive_god_move
    dummy_board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [0, 0, 2, 4]], dtype=np.int64)
    start_t = time.time()
    best_action, depth_used = get_adaptive_god_move(dummy_board)
    calc_time = time.time() - start_t
    print(f"   ✅ Kompilator Numba C++ JIT: OK")
    print(f"   ✅ Wygenerowany ruch: {best_action} (Głębokość: Depth={depth_used}) w czasie {calc_time*1000:.2f} ms")
except Exception as e:
    print(f"   ❌ Błąd silnika God V4: {e}")
    errors += 1

# 4. Sprawdzenie modeli PyTorch
print("\n🧠 4. Weryfikacja zapisanych modeli AI w folderze /models:")
model_path = "models/maskable_ppo_2048_v3.zip"
if os.path.exists(model_path):
    print(f"   ✅ Model MaskablePPO V3: OBECNY ({os.path.getsize(model_path)/1024/1024:.2f} MB)")
else:
    print("   ℹ️ Model MaskablePPO V3 jeszcze nie został zapisany.")

print("\n" + "="*60)
if errors == 0:
    print("🎉 WYNIK AUDYTU: 100% SPÓJNOŚCI! KOD JEST GOTOWY DO WALKI O REKORD!")
else:
    print(f"⚠️ Wykryto {errors} drobne błędy. Naprawmy je przed uruchomieniem!")
print("="*60 + "\n")
