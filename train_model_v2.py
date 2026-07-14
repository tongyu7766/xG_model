import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

df = pd.read_pickle('shots_ff.pkl')

BASE = ['distance', 'angle', 'is_head', 'under_pressure', 'first_time']
FF   = BASE + ['gk_dist_to_goal', 'n_blockers', 'nearest_def_dist']

y = df['goal']
idx_train, idx_test = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y
)

results = {}

for name, feats in [('Base (5 feats)', BASE), ('With freeze-frame (8 feats)', FF)]:
    X_train, X_test = df.loc[idx_train, feats], df.loc[idx_test, feats]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X_train, y_train)
    p = lr.predict_proba(X_test)[:, 1]

    results[name] = {
        'AUC': roc_auc_score(y_test, p),
        'Log loss': log_loss(y_test, p),
        'Brier': brier_score_loss(y_test, p),
        'pred': p, 'model': lr,
    }

    if feats == FF:
        print(f"\n{name} — 逻辑回归系数（标准化后）:")
        for f, c in zip(feats, lr.named_steps['logisticregression'].coef_[0]):
            print(f"  {f:18s}: {c:+.3f}")

# XGBoost 也在全特征上跑一次
X_train, X_test = df.loc[idx_train, FF], df.loc[idx_test, FF]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
xgb.fit(X_train, y_train)
p_xgb = xgb.predict_proba(X_test)[:, 1]
results['XGBoost (8 feats)'] = {
    'AUC': roc_auc_score(y_test, p_xgb),
    'Log loss': log_loss(y_test, p_xgb),
    'Brier': brier_score_loss(y_test, p_xgb),
    'pred': p_xgb,
}

print("\n" + "="*62)
print(f"{'模型':<30}{'AUC':>10}{'LogLoss':>11}{'Brier':>10}")
print("="*62)
for name, r in results.items():
    print(f"{name:<30}{r['AUC']:>10.4f}{r['Log loss']:>11.4f}{r['Brier']:>10.4f}")

print(f"\n实际进球: {y_test.sum()}")
for name in ['Base (5 feats)', 'With freeze-frame (8 feats)', 'XGBoost (8 feats)']:
    print(f"{name}: xG 总和 = {results[name]['pred'].sum():.1f}")

# 校准曲线对比
fig, ax = plt.subplots(figsize=(6, 6))
for name in ['Base (5 feats)', 'With freeze-frame (8 feats)']:
    frac, mean_p = calibration_curve(y_test, results[name]['pred'], n_bins=10, strategy='quantile')
    ax.plot(mean_p, frac, marker='o', label=name)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.set_xlabel('Predicted xG'); ax.set_ylabel('Observed goal rate')
ax.set_title('Calibration: base vs freeze-frame model')
ax.legend(); ax.grid(alpha=0.3)
plt.savefig('calibration_v2.png', dpi=150, bbox_inches='tight')

# 用全量数据重训最终模型并保存（决赛用）
final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
final.fit(df[FF], y)
joblib.dump({'model': final, 'features': FF}, 'xg_model_final.pkl')
print("\n最终模型已保存: xg_model_final.pkl")