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

df = pd.read_pickle('shots_ff2.pkl')

BASE = ['distance', 'angle', 'is_head', 'under_pressure', 'first_time']
FF_OLD = BASE + ['gk_dist_to_goal', 'n_blockers', 'nearest_def_dist']
FF_NEW = BASE + ['gk_dist_to_goal', 'gk_lateral_offset',
                 'n_blockers', 'blocker_density',
                 'nearest_def_dist', 'nearest_def_goalside']

y = df['goal']
# Same random_state as train_model_v2.py -> identical held-out shots, so AUC/
# log loss/Brier deltas below are directly attributable to the feature set,
# not to a different train/test split.
idx_train, idx_test = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(idx_train)}  Test: {len(idx_test)}  "
      f"Test goal rate: {y.loc[idx_test].mean():.3f}")

# ---- multicollinearity check on the richer set (discipline, not just AUC) ----
print("\nFF_NEW feature correlation matrix:")
print(df[FF_NEW].corr().round(2))

results = {}

for name, feats in [('Base (5 feats)', BASE),
                     ('FF old (8 feats)', FF_OLD),
                     ('FF new (11 feats)', FF_NEW)]:
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

    if feats == FF_NEW:
        print(f"\n{name} -- logistic regression coefficients (standardised):")
        for f, c in zip(feats, lr.named_steps['logisticregression'].coef_[0]):
            print(f"  {f:20s}: {c:+.3f}")

# XGBoost on the richer set only, as a ceiling check -- NOT what gets shipped.
# The app markets a "logistic-regression xG model" with inspectable
# coefficients, so the deployed artifact stays LR; XGBoost here just tells us
# whether a flexible model finds much more signal than LR is extracting.
X_train, X_test = df.loc[idx_train, FF_NEW], df.loc[idx_test, FF_NEW]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
xgb.fit(X_train, y_train)
p_xgb = xgb.predict_proba(X_test)[:, 1]
results['XGBoost (11 feats)'] = {
    'AUC': roc_auc_score(y_test, p_xgb),
    'Log loss': log_loss(y_test, p_xgb),
    'Brier': brier_score_loss(y_test, p_xgb),
    'pred': p_xgb,
}

print("\n" + "="*62)
print(f"{'Model':<22}{'AUC':>10}{'LogLoss':>11}{'Brier':>10}")
print("="*62)
for name, r in results.items():
    print(f"{name:<22}{r['AUC']:>10.4f}{r['Log loss']:>11.4f}{r['Brier']:>10.4f}")

print(f"\nActual goals in test set: {y_test.sum()}")
for name in results:
    print(f"{name}: sum(xG) = {results[name]['pred'].sum():.1f}")

# ---- Is FF-new's improvement over FF-old distinguishable from noise? ----
# The test set has only 100 goals, and the AUC/log-loss deltas above are
# small -- report a point estimate without a CI and it's just as easy to be
# chasing sampling noise as a real gain. Paired bootstrap on the same held-out
# shots: resample indices, recompute both models' metrics on that resample.
rng = np.random.default_rng(42)
n_boot = 2000
y_test_arr = y_test.to_numpy()
p_old = results['FF old (8 feats)']['pred']
p_new = results['FF new (11 feats)']['pred']
n = len(y_test_arr)

auc_deltas, ll_deltas = [], []
for _ in range(n_boot):
    sample = rng.integers(0, n, n)
    yb = y_test_arr[sample]
    if yb.sum() == 0 or yb.sum() == n:  # need both classes present for AUC
        continue
    auc_deltas.append(roc_auc_score(yb, p_new[sample]) - roc_auc_score(yb, p_old[sample]))
    ll_deltas.append(log_loss(yb, p_old[sample]) - log_loss(yb, p_new[sample]))  # >0 => new has lower loss

auc_deltas, ll_deltas = np.array(auc_deltas), np.array(ll_deltas)
print(f"\nPaired bootstrap, FF new vs FF old, {len(auc_deltas)} resamples of the same test shots:")
print(f"  AUC delta:     mean {auc_deltas.mean():+.4f}  95% CI [{np.percentile(auc_deltas, 2.5):+.4f}, "
      f"{np.percentile(auc_deltas, 97.5):+.4f}]  P(new > old) = {(auc_deltas > 0).mean():.2f}")
print(f"  LogLoss delta: mean {ll_deltas.mean():+.4f}  95% CI [{np.percentile(ll_deltas, 2.5):+.4f}, "
      f"{np.percentile(ll_deltas, 97.5):+.4f}]  P(new better) = {(ll_deltas > 0).mean():.2f}")

# Calibration curves: base vs old-FF vs new-FF
fig, ax = plt.subplots(figsize=(6, 6))
for name in ['Base (5 feats)', 'FF old (8 feats)', 'FF new (11 feats)']:
    frac, mean_p = calibration_curve(y_test, results[name]['pred'], n_bins=10, strategy='quantile')
    ax.plot(mean_p, frac, marker='o', label=name)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.set_xlabel('Predicted xG'); ax.set_ylabel('Observed goal rate')
ax.set_title('Calibration: base vs old freeze-frame vs new freeze-frame')
ax.legend(); ax.grid(alpha=0.3)
plt.savefig('calibration_v3.png', dpi=150, bbox_inches='tight')
print("\nSaved calibration_v3.png")

# Refit LR on all data for deployment (evaluation numbers above come from the
# untouched held-out split, above -- this refit is only to not throw away 20%
# of training signal in the shipped artifact).
final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
final.fit(df[FF_NEW], y)
joblib.dump({'model': final, 'features': FF_NEW}, 'xg_model_final.pkl')
print("\nFinal model saved: xg_model_final.pkl")
print(f"Features: {FF_NEW}")
