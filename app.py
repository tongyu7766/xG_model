import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

st.set_page_config(page_title="xG Explorer", layout="wide")

GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 8

@st.cache_resource
def load_model():
    bundle = joblib.load('xg_model_final.pkl')
    return bundle['model'], bundle['features']

@st.cache_data
def load_data():
    return pd.read_csv('finishing_table.csv')

model, FEATURES = load_model()

st.title("⚽ Expected Goals (xG) Explorer")
st.caption("Logistic-regression xG model trained on 5,388 open-play shots "
           "from World Cups 2018/2022 and Euros 2020/2024 (StatsBomb open data). "
           "AUC 0.795 · calibrated within ±3% of observed goal rates.")

tab1, tab2 = st.tabs(["🎯 Shot calculator", "📊 Finishing table"])

# ================= TAB 1: 交互计算器 =================
with tab1:
    col_input, col_pitch = st.columns([1, 2])

    with col_input:
        st.subheader("Shot situation")
        x = st.slider("Distance from goal line (x)", 60.0, 120.0, 108.0, 0.5,
                      help="StatsBomb coordinates: goal line at x = 120")
        y = st.slider("Pitch width position (y)", 0.0, 80.0, 40.0, 0.5,
                      help="Centre of pitch = 40; posts at 36 and 44")
        body = st.radio("Body part", ["Foot", "Head"], horizontal=True)
        pressure = st.checkbox("Under pressure from a defender")
        first_time = st.checkbox("First-time shot")
        n_blockers = st.slider("Defenders between ball and goal", 0, 5, 1)
        nearest_def = st.slider("Nearest defender distance (yards)", 0.5, 15.0, 3.0, 0.5)
        gk_pos = st.slider("Goalkeeper distance from goal (yards)", 0.0, 15.0, 1.5, 0.5)

    # 特征构造（与训练完全一致）
    dist = np.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)
    dx, dy = GOAL_X - x, y - GOAL_Y
    ang = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH/2)**2)
    ang = ang + np.pi if ang < 0 else ang

    row = pd.DataFrame([{
        'distance': dist, 'angle': ang,
        'is_head': int(body == "Head"),
        'under_pressure': int(pressure),
        'first_time': int(first_time),
        'gk_dist_to_goal': gk_pos,
        'n_blockers': n_blockers,
        'nearest_def_dist': nearest_def,
    }])
    xg = model.predict_proba(row[FEATURES])[0, 1]

    with col_pitch:
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Goals (xG)", f"{xg:.3f}")
        m2.metric("Distance", f"{dist:.1f} yd")
        m3.metric("Goal angle", f"{np.degrees(ang):.1f}°")

        pitch = VerticalPitch(pitch_type='statsbomb', half=True,
                              pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(7, 5.5))
        fig.set_facecolor('#22312b')
        pitch.scatter([x], [y], s=800 * max(xg, 0.05), c='#e74c3c',
                      edgecolors='white', lw=2, zorder=5, ax=ax)
        # 射门三角
        ax.plot([y, 36], [x, 120], color='white', ls='--', lw=1, alpha=0.5)
        ax.plot([y, 44], [x, 120], color='white', ls='--', lw=1, alpha=0.5)
        ax.set_title(f"xG = {xg:.3f}", color='white', fontsize=16)
        st.pyplot(fig)
        plt.close(fig)

    st.info("**Reading the number:** an xG of 0.10 means the average player scores "
            "this chance about 1 time in 10. Penalties and free kicks are excluded "
            "from the model (different scoring mechanics).")

# ================= TAB 2: 射手榜 =================
with tab2:
    tbl = load_data()
    min_shots = st.slider("Minimum shots to qualify", 5, 40, 15)
    q = tbl[tbl['shots'] >= min_shots].copy()
    q['g_minus_xg'] = q['g_minus_xg'].round(2)
    q['total_xg'] = q['total_xg'].round(2)
    q['xg_per_shot'] = q['xg_per_shot'].round(3)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Outperforming xG")
        st.dataframe(q.nlargest(10, 'g_minus_xg')
                      [['player','shots','goals','total_xg','g_minus_xg']],
                     hide_index=True, use_container_width=True)
    with c2:
        st.subheader("Underperforming xG")
        st.dataframe(q.nsmallest(10, 'g_minus_xg')
                      [['player','shots','goals','total_xg','g_minus_xg']],
                     hide_index=True, use_container_width=True)

    st.warning("**Interpretation note:** goals − xG over four tournaments is a "
               "description of this sample, not proof of finishing skill — the "
               "signal takes several seasons of shots to separate from variance.")