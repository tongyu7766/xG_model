import pandas as pd

df = pd.read_pickle('shots_raw.pkl')

print(f"射门总数: {len(df)}")
print(f"\n字段列表:\n{df.columns.tolist()}")
print(f"\n射门结果分布:\n{df['shot_outcome'].value_counts()}")
print(f"\n射门类型分布:\n{df['shot_type'].value_counts()}")
print(f"\n身体部位分布:\n{df['shot_body_part'].value_counts()}")