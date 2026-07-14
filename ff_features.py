import pandas as pd
import numpy as np

df = pd.read_pickle('shots_clean.pkl')

GOAL_X, GOAL_Y = 120, 40
POST_LEFT = (120, 36)   # 两个门柱
POST_RIGHT = (120, 44)

def sign(o, a, b):
    """点 o 相对于线段 ab 的哪一侧（叉积符号）"""
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def in_triangle(p, v1, v2, v3):
    """点 p 是否在三角形 v1-v2-v3 内"""
    d1, d2, d3 = sign(p, v1, v2), sign(p, v2, v3), sign(p, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def extract_ff_features(row):
    ff = row['shot_freeze_frame']
    shot_loc = (row['x'], row['y'])

    gk_dist_to_goal = np.nan      # 门将到球门中心的距离
    n_blockers = 0                # 射门三角区内的防守球员数（不含门将）
    nearest_def = np.inf          # 最近防守球员距离

    for p in ff:
        if p['teammate']:
            continue
        px, py = p['location']

        is_gk = p['position']['name'] == 'Goalkeeper'
        if is_gk:
            gk_dist_to_goal = np.sqrt((GOAL_X - px)**2 + (GOAL_Y - py)**2)
            continue   # 门将不算 blocker

        # 是否在射门三角区内（射门点 → 两门柱）
        if in_triangle((px, py), shot_loc, POST_LEFT, POST_RIGHT):
            n_blockers += 1

        d = np.sqrt((px - shot_loc[0])**2 + (py - shot_loc[1])**2)
        nearest_def = min(nearest_def, d)

    return pd.Series({
        'gk_dist_to_goal': gk_dist_to_goal,
        'n_blockers': n_blockers,
        'nearest_def_dist': nearest_def if np.isfinite(nearest_def) else np.nan,
    })

print("提取 freeze_frame 特征（约 1 分钟）…")
ff_feats = df.apply(extract_ff_features, axis=1)
df = pd.concat([df, ff_feats], axis=1)

# 缺失处理：门将偶尔不在画面里（比如已被过掉）——那其实是大机会
# 用"门将在门线上"(0) 填充会歪曲事实，这里用中位数填充并另设标记
df['gk_missing'] = df['gk_dist_to_goal'].isna().astype(int)
df['gk_dist_to_goal'] = df['gk_dist_to_goal'].fillna(df['gk_dist_to_goal'].median())
df['nearest_def_dist'] = df['nearest_def_dist'].fillna(df['nearest_def_dist'].median())
# 极端值截断，防止个别远距离值干扰
df['nearest_def_dist'] = df['nearest_def_dist'].clip(upper=30)

# ---- Sanity checks ----
print(f"\n门将缺失的射门: {df['gk_missing'].sum()}")
print(f"\nn_blockers 分布:\n{df['n_blockers'].value_counts().sort_index()}")
print(f"\n按 blockers 数量的进球率:\n{df.groupby(df['n_blockers'].clip(upper=4))['goal'].agg(['mean','count'])}")
print(f"\n门将缺失时的进球率: {df[df['gk_missing']==1]['goal'].mean():.3f}")
print(f"门将在位时的进球率: {df[df['gk_missing']==0]['goal'].mean():.3f}")

df.to_pickle('shots_ff.pkl')
print("\n已保存 shots_ff.pkl")