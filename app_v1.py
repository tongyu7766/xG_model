import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

st.set_page_config(page_title="xG Explorer", layout="wide")

GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 8
POST_LEFT, POST_RIGHT = 36, 44  # y-coordinates of the posts (StatsBomb)

# Mirror ff_features_v2.py exactly -- these constants (and the formulas below)
# must match training-time feature engineering or the app silently feeds the
# model out-of-distribution inputs.
BLOCKER_DIST_FLOOR = 1.0   # yards; caps a single blocker's proximity weight
GK_DIST_CLIP = 25.0        # yards; guards against pathological GK placement
GK_LATERAL_CLIP = 10.0     # yards
NEAREST_DEF_CLIP = 30.0    # yards

@st.cache_resource
def load_model():
    bundle = joblib.load('xg_model_final.pkl')
    return bundle['model'], bundle['features']

@st.cache_data
def load_data():
    return pd.read_csv('finishing_table.csv')

model, FEATURES = load_model()

# ---------- geometry helpers (feature derivation from positions) ----------

def point_in_triangle(p, a, b, c):
    """Barycentric sign test: is point p inside triangle abc?"""
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def derive_defensive_features(shot_xy, defenders, gk_xy):
    """Reproduce the freeze-frame feature engineering used in training
    (ff_features_v2.py), but from user-placed positions.
      n_blockers            : outfield defenders inside the shot cone
                               (triangle shot-location -> both posts)
      blocker_density        : n_blockers weighted by 1/distance to the ball --
                               a defender 1 yard away blocks far more of the
                               goal than one 8 yards away
      nearest_def_dist       : distance from shooter to closest outfield defender
      nearest_def_goalside   : is that nearest defender between the shooter and
                               the goal line, or already beaten?
      gk_dist_to_goal        : GK distance from the centre of the goal line
      gk_lateral_offset      : GK's perpendicular distance off the direct
                               ball-to-goal-centre line -- catches a keeper
                               caught square rather than covering the line
    """
    sx, sy = shot_xy
    shot = np.array(shot_xy, dtype=float)
    post_l = np.array([GOAL_X, POST_LEFT], dtype=float)
    post_r = np.array([GOAL_X, POST_RIGHT], dtype=float)

    n_blockers = 0
    blocker_density = 0.0
    nearest = np.inf
    nearest_goalside = 0
    for _, d in defenders.iterrows():
        p = np.array([d['x'], d['y']], dtype=float)
        dist = np.linalg.norm(p - shot)
        if point_in_triangle(p, shot, post_l, post_r):
            n_blockers += 1
            blocker_density += 1.0 / max(dist, BLOCKER_DIST_FLOOR)
        if dist < nearest:
            nearest = dist
            nearest_goalside = int(d['x'] > sx)
    if not np.isfinite(nearest):
        nearest = 15.0  # no defenders placed -> treat as wide open
        nearest_goalside = 0
    nearest = min(nearest, NEAREST_DEF_CLIP)

    gk_x, gk_y = gk_xy
    gk_dist_to_goal = min(float(np.hypot(GOAL_X - gk_x, GOAL_Y - gk_y)), GK_DIST_CLIP)

    vx, vy = GOAL_X - sx, GOAL_Y - sy
    v_norm = max(np.hypot(vx, vy), 1e-6)
    wx, wy = gk_x - sx, gk_y - sy
    gk_lateral_offset = min(abs(vx * wy - vy * wx) / v_norm, GK_LATERAL_CLIP)

    return (n_blockers, float(nearest), gk_dist_to_goal,
            blocker_density, gk_lateral_offset, nearest_goalside)

# ---------- app ----------

st.title("⚽ Expected Goals (xG) Explorer")
st.caption("Logistic-regression xG model trained on 5,388 open-play shots "
           "from World Cups 2018/2022 and Euros 2020/2024 (StatsBomb open data). "
           "AUC 0.797 · calibrated within ±3% of observed goal rates.")

tab1, tab2 = st.tabs(["🎯 Shot calculator", "📊 Finishing table"])

# ================= TAB 1: positional calculator =================
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

        st.subheader("Defensive setup")
        st.caption("Edit the table to move, add (➕ bottom row) or delete "
                   "defenders. Coordinates use the same system as the shot: "
                   "x = 120 is the goal line, y = 40 is the centre.")
        default_defs = pd.DataFrame({
            'x': [114.0, 112.0],
            'y': [39.0, 43.0],
        })
        defenders = st.data_editor(
            default_defs, num_rows="dynamic", key="defenders",
            column_config={
                'x': st.column_config.NumberColumn("x", min_value=60.0, max_value=120.0, step=0.5),
                'y': st.column_config.NumberColumn("y", min_value=0.0, max_value=80.0, step=0.5),
            },
        )
        defenders = defenders.dropna()

        st.markdown("**Goalkeeper position**")
        gk_x = st.slider("GK x", 105.0, 120.0, 118.5, 0.5)
        gk_y = st.slider("GK y", 30.0, 50.0, 40.0, 0.5)

    # ---- features shared with training ----
    dist = np.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)
    dx, dy = GOAL_X - x, y - GOAL_Y
    ang = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH/2)**2)
    ang = ang + np.pi if ang < 0 else ang

    (n_blockers, nearest_def, gk_dist,
     blocker_density, gk_lateral_offset, nearest_def_goalside) = derive_defensive_features(
        (x, y), defenders, (gk_x, gk_y))

    row = pd.DataFrame([{
        'distance': dist, 'angle': ang,
        'is_head': int(body == "Head"),
        'under_pressure': int(pressure),
        'first_time': int(first_time),
        'gk_dist_to_goal': gk_dist,
        'gk_lateral_offset': gk_lateral_offset,
        'n_blockers': n_blockers,
        'blocker_density': blocker_density,
        'nearest_def_dist': nearest_def,
        'nearest_def_goalside': nearest_def_goalside,
    }])
    xg = model.predict_proba(row[FEATURES])[0, 1]

    with col_pitch:
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Goals (xG)", f"{xg:.3f}")
        m2.metric("Distance", f"{dist:.1f} yd")
        m3.metric("Goal angle", f"{np.degrees(ang):.1f}°")

        d1, d2, d3 = st.columns(3)
        d1.metric("Blockers in cone", n_blockers,
                  help="Outfield defenders inside the shot triangle")
        d2.metric("Nearest defender", f"{nearest_def:.1f} yd")
        d3.metric("GK off goal line", f"{gk_dist:.1f} yd")

        e1, e2, e3 = st.columns(3)
        e1.metric("Blocker density", f"{blocker_density:.2f}",
                  help="Cone blockers weighted by proximity: a defender 1 yd "
                       "away contributes ~8x the weight of one 8 yd away")
        e2.metric("GK lateral offset", f"{gk_lateral_offset:.1f} yd",
                  help="How far the keeper is off the direct ball-to-goal-centre "
                       "line, independent of how deep they're standing")
        e3.metric("Nearest def. goal-side", "Yes" if nearest_def_goalside else "No",
                  help="Is the closest defender shielding the goal, or already beaten?")

        pitch = VerticalPitch(pitch_type='statsbomb', half=True,
                              pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(7, 5.5))
        fig.set_facecolor('#22312b')

        # shot cone (draw first so players sit on top)
        ax.fill([y, POST_LEFT, POST_RIGHT], [x, 120, 120],
                color='white', alpha=0.08, zorder=2)
        ax.plot([y, POST_LEFT], [x, 120], color='white', ls='--', lw=1, alpha=0.5)
        ax.plot([y, POST_RIGHT], [x, 120], color='white', ls='--', lw=1, alpha=0.5)

        # shooter
        pitch.scatter([x], [y], s=800 * max(xg, 0.05), c='#e74c3c',
                      edgecolors='white', lw=2, zorder=5, ax=ax, label='Shooter')

        # defenders: highlight the ones actually blocking the cone
        if len(defenders):
            shot_p = np.array([x, y])
            in_cone = defenders.apply(
                lambda d: point_in_triangle(
                    np.array([d['x'], d['y']]),
                    shot_p, np.array([GOAL_X, POST_LEFT]),
                    np.array([GOAL_X, POST_RIGHT])), axis=1)
            blockers = defenders[in_cone]
            others = defenders[~in_cone]
            if len(blockers):
                pitch.scatter(blockers['x'], blockers['y'], s=250, c='#3498db',
                              edgecolors='white', lw=1.5, zorder=4, ax=ax,
                              label='Blocking defender')
            if len(others):
                pitch.scatter(others['x'], others['y'], s=250, c='#7f8c8d',
                              edgecolors='white', lw=1.5, zorder=4, ax=ax,
                              label='Defender (not blocking)')

        # goalkeeper
        pitch.scatter([gk_x], [gk_y], s=300, c='#f1c40f', marker='s',
                      edgecolors='white', lw=1.5, zorder=4, ax=ax, label='GK')

        ax.set_title(f"xG = {xg:.3f}", color='white', fontsize=16)
        ax.legend(loc='lower left', fontsize=8, facecolor='#22312b',
                  labelcolor='white', framealpha=0.6)
        st.pyplot(fig)
        plt.close(fig)

    st.info("**Reading the number:** an xG of 0.10 means the average player scores "
            "this chance about 1 time in 10. Penalties and free kicks are excluded "
            "from the model (different scoring mechanics). Blue defenders sit inside "
            "the shot cone and count as blockers; grey ones only matter through the "
            "nearest-defender distance and whether they're still goal-side. Blocker "
            "density weights cone defenders by proximity, so one defender standing "
            "a yard off the shooter outweighs several loitering at range. GK lateral "
            "offset flags a keeper caught square rather than covering the near post.")

# ================= TAB 2: finishing table (unchanged) =================
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