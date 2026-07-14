from statsbombpy import sb
import pandas as pd

pd.set_option('display.max_rows', 200)

# 列出所有免费可用的赛事
comps = sb.competitions()
print(comps[['competition_id', 'season_id', 'competition_name', 'season_name']].to_string())