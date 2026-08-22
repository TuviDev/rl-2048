# 🧠 Deep Reinforcement Learning 2048 Solver

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.4-orange.svg)
![RL](https://img.shields.io/badge/Algorithm-MaskablePPO%20%2B%20Expectimax-green.svg)
![Framework](https://img.shields.io/badge/Gymnasium-v1.0-brightgreen.svg)

A high-performance **Hybrid Artificial Intelligence System** trained to master the game **2048**. Combines Convolutional Neural Networks (CNNs), Action Masking, and Expectimax Tree Search.

---

## 📊 Benchmark Results

| Model Architecture | Avg Score | Max Tile Reached | Sample Efficiency / FPS |
| :--- | :---: | :---: | :---: |
| **Random Player** | 350 | 64 | N/A |
| **Baseline PPO (MLP)** | 1,800 | 256 | ~800 FPS |
| **Maskable PPO (CNN)** | **4,800** | **512** | **~1,600 FPS** |
| **Hybrid CNN + Expectimax (Depth=3)** | **8,500+** | **1024 / 2048** | **Real-time (0.03s/move)** |

---

## 🚀 Quick Start

### 1. Clone & Activate Environment
```bash
git clone https://github.com/YOUR_USERNAME/rl-2048.git
cd rl-2048
python -m venv venv