import streamlit as st
import numpy as np
import time
import torch
import plotly.express as px
from sb3_contrib import MaskablePPO
from game2048_v3_mask import Game2048V3MaskEnv
from evaluate_ultra_fast import get_best_move

st.set_page_config(page_title="AI 2048 Master Dashboard", page_icon="🎮", layout="wide")

TILE_COLORS = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32"
}

TEXT_COLORS = {0: "#cdc1b4", 2: "#776e65", 4: "#776e65"}

def render_board_html(board):
    html = "<div style='display: grid; grid-template-columns: repeat(4, 90px); gap: 12px; background-color: #bbada0; padding: 12px; border-radius: 12px; width: fit-content; margin: auto; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>"
    for r in range(4):
        for c in range(4):
            val = int(board[r, c])
            bg = TILE_COLORS.get(val, "#3c3a32")
            color = TEXT_COLORS.get(val, "#f9f6f2")
            font_size = "30px" if val < 100 else ("24px" if val < 1000 else "20px")
            text = str(val) if val > 0 else ""
            html += f"<div style='background-color: {bg}; color: {color}; font-weight: bold; font-size: {font_size}; display: flex; align-items: center; justify-content: center; height: 90px; border-radius: 6px;'>{text}</div>"
    html += "</div>"
    return html

st.title("🧠 Deep Reinforcement Learning: 2048 Solver")
st.markdown("Aplikacja demonstracyjna ukazująca działanie **MaskablePPO (CNN)** oraz **Expectimax Tree Search**.")

col_left, col_right = st.columns([1.2, 0.8])

with col_right:
    st.subheader("⚙️ Panel Sterowania AI")
    ai_mode = st.selectbox(
        "Wybierz Silnik AI:",
        ["⚡ Hybrid AI (Expectimax Depth=3)", "🤖 MaskablePPO (Czysta Sieć V3)", "🎲 Losowy Gracz"]
    )
    speed = st.slider("Opóźnienie ruchu (sekund):", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    start_button = st.button("🚀 Uruchom Mecz AI", type="primary", use_container_width=True)

with col_left:
    st.subheader("🕹️ Gra na Żywo")
    board_placeholder = st.empty()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    score_metric = metric_col1.empty()
    tile_metric = metric_col2.empty()
    moves_metric = metric_col3.empty()
    chart_placeholder = st.empty()

if start_button:
    env = Game2048V3MaskEnv()
    obs, info = env.reset()
    done = False
    step = 0
    scores_history = [0]
    
    # Ładowanie modelu MaskablePPO jeśli wybrano
    model_mask = None
    if "MaskablePPO" in ai_mode:
        try:
            model_mask = MaskablePPO.load("models/maskable_ppo_2048_v3")
        except Exception:
            st.warning("⚠️ Model V3 jeszcze się nie wytrenował. Uruchom 'python train_v3_mask.py'!")

    while not done:
        if "Hybrid AI" in ai_mode:
            action = get_best_move(env.board, depth=3)
        elif "MaskablePPO" in ai_mode and model_mask is not None:
            masks = env.action_masks()
            action, _ = model_mask.predict(obs, action_masks=masks)
        else:
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

        obs, reward, done, truncated, info = env.step(action)
        step += 1
        scores_history.append(info["score"])

        # Renderowanie planszy i metryk
        board_placeholder.markdown(render_board_html(env.board), unsafe_allow_html=True)
        score_metric.metric("Wynik (Score)", f"{info['score']:,}")
        tile_metric.metric("Max Kafelek", f"{info['max_tile']}")
        moves_metric.metric("Liczba Ruchów", f"{step}")

        time.sleep(speed)

    st.balloons()
    st.success(f"🏁 Koniec Gry! Wynik Końcowy: {info['score']:,} | Najwyższy Kafelek: {info['max_tile']}")
