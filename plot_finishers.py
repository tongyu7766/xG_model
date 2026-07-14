import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tbl = pd.read_csv('finishing_table.csv')

MIN_SHOTS = 15
q = tbl[tbl['shots'] >= MIN_SHOTS].copy()

# 要标注名字的球员（用姓氏短名，图上不挤）
LABELS = {
    'Harry Kane': 'Kane',
    'Kylian Mbappé Lottin': 'Mbappé',
    'Cody Mathès Gakpo': 'Gakpo',
    'Xherdan Shaqiri': 'Shaqiri',
    'Neymar da Silva Santos Junior': 'Neymar',
    'Cristiano Ronaldo dos Santos Aveiro': 'Ronaldo',
    'Antoine Griezmann': 'Griezmann',
    'Marcus Berg': 'Berg',
    'Lionel Andrés Messi Cuccittini': 'Messi',
    'Romelu Lukaku Menama': 'Lukaku',
    'Ivan Perišić': 'Perišić',
    'Bukayo Saka': 'Saka',
}

fig, ax = plt.subplots(figsize=(9, 8))

# 对角线：goals = xG（完美符合模型预期）
lim = max(q['total_xg'].max(), q['goals'].max()) + 1
ax.plot([0, lim], [0, lim], '--', color='#7f8c8d', lw=1.5,
        label='Goals = xG (as expected)', zorder=1)

# 上下着色区
ax.fill_between([0, lim], [0, lim], [lim, lim],
                color='#e74c3c', alpha=0.05, zorder=0)
ax.fill_between([0, lim], [0, 0], [0, lim],
                color='#3498db', alpha=0.05, zorder=0)
ax.text(0.6, lim - 0.8, 'Outperforming xG', color='#c0392b',
        fontsize=10, style='italic')
ax.text(lim - 3.4, 0.4, 'Underperforming xG', color='#2c6e9e',
        fontsize=10, style='italic')

# 散点：颜色 = 超额方向，大小 = 射门量
over = q['g_minus_xg'] >= 0
ax.scatter(q.loc[over, 'total_xg'], q.loc[over, 'goals'],
           s=q.loc[over, 'shots'] * 4, c='#e74c3c',
           edgecolors='white', lw=0.8, alpha=0.8, zorder=3)
ax.scatter(q.loc[~over, 'total_xg'], q.loc[~over, 'goals'],
           s=q.loc[~over, 'shots'] * 4, c='#3498db',
           edgecolors='white', lw=0.8, alpha=0.8, zorder=3)

# 标注重点球员
# 每个球员单独的标注偏移 (x, y)，负值往左/下
OFFSETS = {
    'Kane': (-38, 8), 'Mbappé': (10, -2),
    'Gakpo': (-42, -4), 'Shaqiri': (8, 8),
    'Perišić': (8, 8), 'Saka': (-34, 8),
    'Messi': (8, -14), 'Lukaku': (10, 4),
    'Ronaldo': (10, 4), 'Neymar': (10, 4),
    'Griezmann': (8, 8), 'Berg': (8, 8),
}

for full, short in LABELS.items():
    row = q[q['player'] == full]
    if row.empty:
        continue
    x, y = row['total_xg'].iloc[0], row['goals'].iloc[0]
    dx, dy = OFFSETS.get(short, (6, 6))
    ax.annotate(short, (x, y),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#2c3e50')
    
ax.set_xlabel('Total xG (model-expected goals)', fontsize=12)
ax.set_ylabel('Actual goals', fontsize=12)
ax.set_title(f'Finishing vs expectation — World Cups 2018/22 & Euros 2020/24\n'
             f'(open-play shots, players with {MIN_SHOTS}+ shots; '
             f'marker size = shot volume)', fontsize=12.5, pad=12)
ax.set_xlim(0, lim); ax.set_ylim(-0.3, lim)
ax.grid(alpha=0.25)
ax.legend(loc='upper left', fontsize=10)

plt.savefig('finishing_scatter.png', dpi=150, bbox_inches='tight')
print("已保存 finishing_scatter.png")