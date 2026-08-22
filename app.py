import streamlit as st
import numpy as np
import time
from game2048_v3_mask import Game2048V3MaskEnv
from expectimax_god_engine import get_god_move, clear_transposition_table

st.set_page_config(page_title="2048 AI Dashboard", page_icon="🎮", layout="wide")

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

st.title("🧠 2048 AI Solver: Bitboard & Expectimax")
st.markdown("Aplikacja demonstrująca działanie sztucznej inteligencji opartej na **64-bitowych rejestrach CPU** i **Expectimax Tree Search**.")

col_left, col_right = st.columns([1.2, 0.8])

with col_right:
    st.subheader("⚙️ Ustawienia")
    speed = st.slider("Opóźnienie animacji (sekund):", min_value=0.01, max_value=0.5, value=0.03, step=0.01)
    start_button = st.button("🚀 Uruchom Mecz AI", type="primary", use_container_width=True)

with col_left:
    st.subheader("🕹️ Rozgrywka Na Żywo")
    board_placeholder = st.empty()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    score_metric = metric_col1.empty()
    tile_metric = metric_col2.empty()
    moves_metric = metric_col3.empty()

if start_button:
    clear_transposition_table()
    env = Game2048V3MaskEnv()
    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        masks = env.action_masks()
        action, depth_used = get_god_move(env.board, valid_mask=masks)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        board_placeholder.markdown(render_board_html(env.board), unsafe_allow_html=True)
        score_metric.metric("Wynik (Score)", f"{info['score']:,}")
        tile_metric.metric("Max Kafelek", f"{info['max_tile']}")
        moves_metric.metric("Liczba Ruchów", f"{step}")

        time.sleep(speed)

    st.balloons()
    st.success(f"🏁 Koniec Gry! Wynik Końcowy: {info['score']:,} | Najwyższy Kafelek: {info['max_tile']}")
