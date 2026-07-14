import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

df = pd.read_pickle('shots_clean.pkl')

FEATURES = ['distance', 'angle', 'is_head', 'under_pressure', 'first_time']
X = df[FEATURES]
y = df['goal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ---------- 模型 1：逻辑回归（基线） ----------
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)
p_lr = logreg.predict_proba(X_test)[:, 1]

# 看系数（可解释性 —— 面试考点）
coefs = logreg.named_steps['logisticregression'].coef_[0]
print("\n逻辑回归系数（标准化后）:")
for f, c in zip(FEATURES, coefs):
    print(f"  {f:16s}: {c:+.3f}")

# ---------- 模型 2：XGBoost ----------
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'
)
xgb.fit(X_train, y_train)
p_xgb = xgb.predict_proba(X_test)[:, 1]

# ---------- 评估 ----------
print("\n" + "="*50)
print(f"{'指标':<14}{'LogReg':>12}{'XGBoost':>12}")
print("="*50)
for metric, fn in [('ROC-AUC', roc_auc_score),
                   ('Log loss', log_loss),
                   ('Brier score', brier_score_loss)]:
    print(f"{metric:<14}{fn(y_test, p_lr):>12.4f}{fn(y_test, p_xgb):>12.4f}")

# 关键验证：xG 总和 应 ≈ 实际进球数
print("\n--- xG 总和 vs 实际进球 ---")
print(f"实际进球数    : {y_test.sum()}")
print(f"LogReg  预测  : {p_lr.sum():.1f}")
print(f"XGBoost 预测  : {p_xgb.sum():.1f}")

# ---------- 校准曲线 ----------
fig, ax = plt.subplots(figsize=(6, 6))
for name, p in [('Logistic Regression', p_lr), ('XGBoost', p_xgb)]:
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy='quantile')
    ax.plot(mean_pred, frac_pos, marker='o', label=name)
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.set_xlabel('Predicted xG')
ax.set_ylabel('Observed goal rate')
ax.set_title('Calibration curve')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig('calibration.png', dpi=150, bbox_inches='tight')
print("\n已保存 calibration.png")