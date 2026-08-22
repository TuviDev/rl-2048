import streamlit as st
import numpy as np
import time
from game2048_v3_mask import Game2048V3MaskEnv
from bitboard_v5_god import get_v5_ultimate_move
from expectimax_god_v4 import get_adaptive_god_move

st.set_page_config(page_title="AI 2048 Master Dashboard", page_icon="🎮", layout="wide")

TILE_COLORS = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32"
}

TEXT_COLORS = {0: "#cdc1b4", 2: "#776e65", 4: "#776e65"}

def render_board_html(board):
    html = "<div style='display: grid; grid-template-columns: repeat(4, 95px); gap: 12px; background-color: #bbada0; padding: 14px; border-radius: 12px; width: fit-content; margin: auto; box-shadow: 0 10px 20px rgba(0,0,0,0.4);'>"
    for r in range(4):
        for c in range(4):
            val = int(board[r, c])
            bg = TILE_COLORS.get(val, "#3c3a32")
            color = TEXT_COLORS.get(val, "#f9f6f2")
            font_size = "32px" if val < 100 else ("26px" if val < 1000 else "20px")
            text = str(val) if val > 0 else ""
            html += f"<div style='background-color: {bg}; color: {color}; font-weight: bold; font-size: {font_size}; display: flex; align-items: center; justify-content: center; height: 95px; border-radius: 8px;'>{text}</div>"
    html += "</div>"
    return html

st.title("🧠 Deep Reinforcement Learning & Bitboard 2048 Solver")
st.markdown("Interaktywna Aplikacja AI oparta na **64-bitowym Bitboardzie C++ ($O(1)$ Lookup)** oraz **Expectimax D4 Symmetry**.")

col_left, col_right = st.columns([1.2, 0.8])

with col_right:
    st.subheader("⚙️ Ustawienia Silnika AI")
    ai_mode = st.selectbox(
        "Wybierz Silnik AI:",
        ["⚡ Bitboard V5 God Engine (O(1) C++)", "👑 Expectimax V4 (D4 Symmetry)", "🎲 Losowy Gracz"]
    )
    speed = st.slider("Opóźnienie animacji (sekundy):", min_value=0.01, max_value=0.5, value=0.04, step=0.01)
    start_button = st.button("🚀 Uruchom Mecz AI", type="primary", use_container_width=True)

with col_left:
    st.subheader("🕹️ Rozgrywka Na Żywo")
    board_placeholder = st.empty()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    score_metric = metric_col1.empty()
    tile_metric = metric_col2.empty()
    moves_metric = metric_col3.empty()

if start_button:
    env = Game2048V3MaskEnv()
    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        masks = env.action_masks()
        
        if "Bitboard V5" in ai_mode:
            action, depth_used = get_v5_ultimate_move(env.board, valid_mask=masks)
        elif "Expectimax V4" in ai_mode:
            action, depth_used = get_adaptive_god_move(env.board, valid_mask=masks)
        else:
            valid_actions = np.where(masks)[0]
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

        obs, reward, done, truncated, info = env.step(action)
        step += 1

        board_placeholder.markdown(render_board_html(env.board), unsafe_allow_html=True)
        score_metric.metric("Wynik (Score)", f"{info['score']:,}")
        tile_metric.metric("Max Kafelek", f"{info['max_tile']}")
        moves_metric.metric("Liczba Ruchów", f"{step}")

        time.sleep(speed)

    st.balloons()
    st.success(f"🏁 Koniec Gry! Wynik Końcowy: {info['score']:,} | Najwyższy Kafelek: {info['max_tile']}")
