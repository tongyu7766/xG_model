import pandas as pd
import numpy as np

df = pd.read_pickle('shots_clean.pkl')

GOAL_X, GOAL_Y = 120, 40
POST_LEFT = (120, 36)
POST_RIGHT = (120, 44)

# Below this distance, a blocker's proximity weight is capped at 1.0 instead of
# blowing up toward infinity for a defender standing ~on top of the shooter.
BLOCKER_DIST_FLOOR = 1.0


def sign(o, a, b):
    """Which side of segment ab point o is on (cross-product sign)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def in_triangle(p, v1, v2, v3):
    """Is point p inside triangle v1-v2-v3?"""
    d1, d2, d3 = sign(p, v1, v2), sign(p, v2, v3), sign(p, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def extract_ff_features(row):
    ff = row['shot_freeze_frame']
    sx, sy = row['x'], row['y']
    shot_loc = (sx, sy)

    gk_dist_to_goal = np.nan
    gk_lateral_offset = np.nan
    n_blockers = 0
    blocker_density = 0.0
    nearest_def = np.inf
    nearest_def_goalside = np.nan

    # Direction from the shot to goal centre -- the line a keeper "should" be
    # on. Used to project the GK's position and measure how far off it they are.
    vx, vy = GOAL_X - sx, GOAL_Y - sy
    v_norm = max(np.hypot(vx, vy), 1e-6)

    for p in ff:
        if p['teammate']:
            continue
        px, py = p['location']

        if p['position']['name'] == 'Goalkeeper':
            gk_dist_to_goal = np.hypot(GOAL_X - px, GOAL_Y - py)
            # Perpendicular distance from the GK to the shot->goal-centre line
            # (cross product of v and (K-S), normalised by |v|). Separates "off
            # their line sideways" from "standing deep/off gk_dist_to_goal" --
            # a keeper can be close to goal-centre-distance but badly split wide.
            wx, wy = px - sx, py - sy
            gk_lateral_offset = abs(vx * wy - vy * wx) / v_norm
            continue  # keeper is never counted as a blocker

        d = np.hypot(px - sx, py - sy)

        if in_triangle((px, py), shot_loc, POST_LEFT, POST_RIGHT):
            n_blockers += 1
            # Closer defenders inside the cone block more of the goal mouth;
            # weight their contribution by inverse distance instead of counting
            # everyone in the cone equally.
            blocker_density += 1.0 / max(d, BLOCKER_DIST_FLOOR)

        if d < nearest_def:
            nearest_def = d
            # "Goal-side" = between the shooter and the goal line (StatsBomb
            # x=120 is the goal), the standard defensive-positioning proxy.
            # A near defender who is NOT goal-side has already been beaten and
            # isn't shielding the shot the way nearest_def_dist alone implies.
            nearest_def_goalside = int(px > sx)

    return pd.Series({
        'gk_dist_to_goal': gk_dist_to_goal,
        'gk_lateral_offset': gk_lateral_offset,
        'n_blockers': n_blockers,
        'blocker_density': blocker_density,
        'nearest_def_dist': nearest_def if np.isfinite(nearest_def) else np.nan,
        'nearest_def_goalside': nearest_def_goalside,
    })


print("Extracting freeze-frame features...")
ff_feats = df.apply(extract_ff_features, axis=1)
df = pd.concat([df, ff_feats], axis=1)

# Missing-data handling (same discipline as v1): a missing GK means the
# keeper's already been beaten and isn't in frame -- that's a big chance, not
# "keeper on the line" -- so it gets a flag rather than a value of 0.
df['gk_missing'] = df['gk_dist_to_goal'].isna().astype(int)
df['gk_dist_to_goal'] = df['gk_dist_to_goal'].fillna(df['gk_dist_to_goal'].median())
df['gk_lateral_offset'] = df['gk_lateral_offset'].fillna(df['gk_lateral_offset'].median())

df['nearest_def_missing'] = df['nearest_def_dist'].isna().astype(int)
df['nearest_def_dist'] = df['nearest_def_dist'].fillna(df['nearest_def_dist'].median())
df['nearest_def_dist'] = df['nearest_def_dist'].clip(upper=30)
df['nearest_def_goalside'] = df['nearest_def_goalside'].fillna(df['nearest_def_goalside'].median())

# A few freeze frames (~9 of 5388) tag the "Goalkeeper" 20-65 yards from their
# own goal -- bad position labels, not real sweeper-keeper play (no keeper
# plays that far out in open play). v1 already clipped nearest_def_dist for
# this exact reason but never applied it to the GK features; a handful of
# these were bad enough to send gk_lateral_offset as high as 52 yards, which
# would have dominated StandardScaler's mean/std. Same clip discipline here.
df['gk_dist_to_goal'] = df['gk_dist_to_goal'].clip(upper=25)
df['gk_lateral_offset'] = df['gk_lateral_offset'].clip(upper=10)

# ---- Sanity checks ----
print(f"\nShots with GK missing from frame: {df['gk_missing'].sum()}")
print(f"Shots with no outfield defender in frame: {df['nearest_def_missing'].sum()}")

print(f"\nblocker_density describe:\n{df['blocker_density'].describe()}")
print(f"\ngk_lateral_offset describe:\n{df['gk_lateral_offset'].describe()}")

print(f"\ncorr(n_blockers, blocker_density)            = {df['n_blockers'].corr(df['blocker_density']):.3f}")
print(f"corr(nearest_def_dist, nearest_def_goalside) = {df['nearest_def_dist'].corr(df['nearest_def_goalside']):.3f}")
print(f"corr(gk_dist_to_goal, gk_lateral_offset)      = {df['gk_dist_to_goal'].corr(df['gk_lateral_offset']):.3f}")

print(f"\nGoal rate, nearest defender goal-side:     {df[df['nearest_def_goalside']==1]['goal'].mean():.3f}"
      f"  (n={ (df['nearest_def_goalside']==1).sum() })")
print(f"Goal rate, nearest defender NOT goal-side: {df[df['nearest_def_goalside']==0]['goal'].mean():.3f}"
      f"  (n={ (df['nearest_def_goalside']==0).sum() })")

print(f"\nGoal rate by blocker_density quartile:\n"
      f"{df.groupby(pd.qcut(df['blocker_density'], 4, duplicates='drop'), observed=True)['goal'].agg(['mean','count'])}")
print(f"\nGoal rate by gk_lateral_offset quartile:\n"
      f"{df.groupby(pd.qcut(df['gk_lateral_offset'], 4, duplicates='drop'), observed=True)['goal'].agg(['mean','count'])}")

df.to_pickle('shots_ff2.pkl')
print("\nSaved shots_ff2.pkl")
