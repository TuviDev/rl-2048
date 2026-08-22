# 🗺️ 2048 AI Solver — Future Technical Roadmap

Gdybyś chciał w przyszłości rozwinąć ten projekt jeszcze dalej, oto sprawdzona ścieżka:

---

## �� Krok 1: Wdrożenie C++ DLL / Cython Extensions
* **Opis:** Przepisanie funkcji `expectimax` z Numba JIT bezpośrednio do natywnego pliku `.dll` / `.so` skompilowanego w C++17 (z flagami `-O3 -march=native`).
* **Zysk:** Dostęp do instrukcji wektorowych AVX-512 oraz całkowita eliminacja narzutu pamięci Pythona.
* **Cel:** 100 000 000 sprawdzonych stanów/sek.

## 🟡 Krok 2: Optymalizacja Wag Heurystycznych przez CMA-ES
* **Opis:** Zastosowanie algorytmu ewolucyjnego CMA-ES (Covariance Matrix Adaptation Evolution Strategy) na klastrze wieloprocesorowym, aby automatycznie dobrać wagi gładkości, monotoniczności i pustych pól.
* **Zysk:** Eliminacja ludzkiego zgadywania wag.
* **Cel:** Stabilne osiąganie kafelka 8192 i 16384.

## 🔴 Krok 3: Export do WebAssembly (WASM) / Czystego JS
* **Opis:** Skompilowanie silnika C++ do WebAssembly (WASM) za pomocą Emscriptena.
* **Zysk:** Możliwość uruchomienia bota bezpośrednio w przeglądarce klienta bez żadnego backendu serwerowego.
* **Cel:** Darmowy hosting dla tysięcy graczy.
