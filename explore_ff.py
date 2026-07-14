import pandas as pd

df = pd.read_pickle('shots_clean.pkl')

print(f"有 freeze_frame 的射门: {df['shot_freeze_frame'].notna().sum()} / {len(df)}")

# 看一条的结构
ff = df.loc[df['shot_freeze_frame'].notna(), 'shot_freeze_frame'].iloc[0]
print(f"\n类型: {type(ff)}, 元素数: {len(ff)}")
print("\n第一个元素:")
print(ff[0])
print("\n找守门员:")
for p in ff:
    if p['position']['name'] == 'Goalkeeper' and not p['teammate']:
        print(p)