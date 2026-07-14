import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch, Pitch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_pickle('shots_clean.pkl')
FEATURES = ['distance', 'angle', 'is_head', 'under_pressure', 'first_time']

# 用全部数据重新训练，并保存模型（决赛当晚要用）
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(df[FEATURES], df['goal'])
joblib.dump(model, 'xg_model.pkl')
print("模型已保存: xg_model.pkl")

df['xg'] = model.predict_proba(df[FEATURES])[:, 1]

# ============ 图 1：射门地图 ============
pitch = VerticalPitch(pitch_type='statsbomb', half=True,
                      pitch_color='#22312b', line_color='#c7d5cc')
fig, ax = pitch.draw(figsize=(9, 7))
fig.set_facecolor('#22312b')

goals = df[df['goal'] == 1]
misses = df[df['goal'] == 0]

pitch.scatter(misses['x'], misses['y'], s=misses['xg'] * 400,
              c='#7f8c8d', edgecolors='#2c3e50', alpha=0.35,
              label='No goal', ax=ax)
pitch.scatter(goals['x'], goals['y'], s=goals['xg'] * 400,
              c='#e74c3c', edgecolors='white', alpha=0.85,
              label='Goal', ax=ax)

ax.legend(facecolor='#22312b', edgecolor='None', fontsize=11,
          labelcolor='white', loc='upper left')
ax.set_title('Open-play shots — World Cup & Euros\n(marker size = xG)',
             color='white', fontsize=15, pad=12)
plt.savefig('shot_map.png', dpi=150, bbox_inches='tight', facecolor='#22312b')
plt.close()
print("已保存 shot_map.png")

# ============ 图 2：xG 曲面（模型学到了什么） ============
GOAL_X, GOAL_Y, GOAL_WIDTH = 120, 40, 8

xs = np.linspace(60, 120, 120)
ys = np.linspace(0, 80, 160)
XX, YY = np.meshgrid(xs, ys)

dist = np.sqrt((GOAL_X - XX)**2 + (GOAL_Y - YY)**2)
dx = GOAL_X - XX
dy = YY - GOAL_Y
ang = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2)**2)
ang = np.where(ang < 0, ang + np.pi, ang)

grid = pd.DataFrame({
    'distance': dist.ravel(),
    'angle': ang.ravel(),
    'is_head': 0,            # 假设：右脚、无逼抢、非一脚射
    'under_pressure': 0,
    'first_time': 0,
})
ZZ = model.predict_proba(grid[FEATURES])[:, 1].reshape(XX.shape)

pitch2 = Pitch(pitch_type='statsbomb', half=True,
               pitch_color='none', line_color='black', linewidth=1.5)
fig, ax = pitch2.draw(figsize=(10, 7))

cs = ax.contourf(XX, YY, ZZ, levels=20, cmap='hot', alpha=0.75)
cbar = fig.colorbar(cs, ax=ax, shrink=0.75)
cbar.set_label('Expected Goals (xG)', fontsize=11)
ax.set_title('xG surface — probability of scoring by shot location\n'
             '(foot shot, no pressure)', fontsize=14, pad=12)
plt.savefig('xg_surface.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 xg_surface.png")

# ============ 图 3：距离 vs xG（模型 vs 实际） ============
fig, ax = plt.subplots(figsize=(8, 5))

bins = np.arange(0, 45, 3)
df['dbin'] = pd.cut(df['distance'], bins)
actual = df.groupby('dbin', observed=True)['goal'].mean()
predicted = df.groupby('dbin', observed=True)['xg'].mean()
centers = [b.mid for b in actual.index]

ax.plot(centers, actual, 'o-', label='Observed goal rate', color='#e74c3c', lw=2)
ax.plot(centers, predicted, 's--', label='Model xG', color='#3498db', lw=2)
ax.set_xlabel('Distance to goal (yards)')
ax.set_ylabel('Probability of scoring')
ax.set_title('Model fit across distance')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('distance_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print("已保存 distance_fit.png")