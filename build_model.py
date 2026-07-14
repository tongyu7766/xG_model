import pandas as pd
import numpy as np

df = pd.read_pickle('shots_raw.pkl')
print(f"原始射门数: {len(df)}")

# ---------- 1. 清洗 ----------
# 只保留运动战射门：点球、任意球、角球的进球机制完全不同
df = df[df['shot_type'] == 'Open Play'].copy()

# 目标变量
df['goal'] = (df['shot_outcome'] == 'Goal').astype(int)

# 拆坐标：StatsBomb 球场 120 x 80（码），球门在 x=120，y 从 36 到 44
df = df[df['location'].notna()].copy()
df[['x', 'y']] = pd.DataFrame(df['location'].tolist(), index=df.index)

print(f"运动战射门: {len(df)}, 进球: {df['goal'].sum()}, 进球率: {df['goal'].mean():.3f}")

# ---------- 2. 特征工程 ----------
GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 8

# 到球门中心的距离
df['distance'] = np.sqrt((GOAL_X - df['x'])**2 + (GOAL_Y - df['y'])**2)

# 可视角度：球门在射门点张开的弧度（xG 最强的特征之一）
dx = GOAL_X - df['x']
dy = df['y'] - GOAL_Y
angle = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2)**2)
df['angle'] = np.where(angle < 0, angle + np.pi, angle)

# 身体部位
df['is_head'] = (df['shot_body_part'] == 'Head').astype(int)

# 是否被逼抢
df['under_pressure'] = df['under_pressure'].fillna(False).astype(int)

# 是否一脚射门（不停球直接打）
if 'shot_first_time' in df.columns:
    df['first_time'] = df['shot_first_time'].fillna(False).astype(int)
else:
    df['first_time'] = 0

FEATURES = ['distance', 'angle', 'is_head', 'under_pressure', 'first_time']

# 检查特征是否合理
print(f"\n特征描述:\n{df[FEATURES].describe()}")

# 关键 sanity check：距离越近、角度越大，进球率应该越高
df['dist_bin'] = pd.cut(df['distance'], bins=[0, 6, 12, 18, 25, 100])
print(f"\n各距离区间的进球率:\n{df.groupby('dist_bin', observed=True)['goal'].agg(['mean', 'count'])}")

df.to_pickle('shots_clean.pkl')
print("\n已保存 shots_clean.pkl")