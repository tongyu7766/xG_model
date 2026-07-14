from statsbombpy import sb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 要拉的赛事：(competition_id, season_id, 名称)
COMPETITIONS = [
    (43, 106, 'World Cup 2022'),
    (43, 3,   'World Cup 2018'),
    (55, 282, 'Euro 2024'),
    (55, 43,  'Euro 2020'),
]

all_shots = []

for comp_id, season_id, name in COMPETITIONS:
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    print(f"{name}: {len(matches)} 场比赛，开始拉取…")

    for i, mid in enumerate(matches['match_id'], 1):
        try:
            ev = sb.events(match_id=mid)
            shots = ev[ev['type'] == 'Shot'].copy()
            shots['competition'] = name
            all_shots.append(shots)
        except Exception as e:
            print(f"  第 {mid} 场失败: {e}")

        if i % 10 == 0:
            print(f"  已完成 {i}/{len(matches)}")

df = pd.concat(all_shots, ignore_index=True)

print(f"\n=== 总射门数: {len(df)} ===")
print(f"\n字段列表:\n{df.columns.tolist()}")
print(f"\n射门结果分布:\n{df['shot_outcome'].value_counts()}")
print(f"\n射门类型分布:\n{df['shot_type'].value_counts()}")
print(f"\n身体部位分布:\n{df['shot_body_part'].value_counts()}")

df.to_pickle('shots_raw.pkl')
print("\n已保存到 shots_raw.pkl")