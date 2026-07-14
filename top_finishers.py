import pandas as pd
import numpy as np
import joblib

df = pd.read_pickle('shots_ff.pkl')
bundle = joblib.load('xg_model_final.pkl')
model, FEATURES = bundle['model'], bundle['features']

# 用最终模型给每次射门打分
df['xg'] = model.predict_proba(df[FEATURES])[:, 1]

# ---------- 按球员汇总 ----------
tbl = (df.groupby('player')
         .agg(shots=('xg', 'count'),
              goals=('goal', 'sum'),
              total_xg=('xg', 'sum'))
         .reset_index())

tbl['g_minus_xg'] = tbl['goals'] - tbl['total_xg']
tbl['xg_per_shot'] = tbl['total_xg'] / tbl['shots']

# 关键过滤：射门太少的球员差值全是噪声
MIN_SHOTS = 15
qualified = tbl[tbl['shots'] >= MIN_SHOTS].copy()
print(f"达到 {MIN_SHOTS} 次射门门槛的球员: {len(qualified)} / {len(tbl)}")

cols = ['player', 'shots', 'goals', 'total_xg', 'g_minus_xg', 'xg_per_shot']

print("\n========= 射术超额榜（Goals − xG 最高）=========")
print(qualified.nlargest(10, 'g_minus_xg')[cols].round(2).to_string(index=False))

print("\n========= 射术欠额榜（Goals − xG 最低）=========")
print(qualified.nsmallest(10, 'g_minus_xg')[cols].round(2).to_string(index=False))

# 顺手：按球队汇总（各队创造机会质量 vs 转化）
team_tbl = (df.groupby('team')
              .agg(shots=('xg', 'count'),
                   goals=('goal', 'sum'),
                   total_xg=('xg', 'sum'))
              .reset_index())
team_tbl['g_minus_xg'] = team_tbl['goals'] - team_tbl['total_xg']

print("\n========= 球队榜：转化超额 Top 8 =========")
print(team_tbl.nlargest(8, 'g_minus_xg').round(2).to_string(index=False))

tbl.to_csv('finishing_table.csv', index=False)
print("\n已保存 finishing_table.csv")