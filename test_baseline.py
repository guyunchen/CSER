import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. 加载数据
# ===============================
PATH = "dataset/IEMOCAP/random_data/train_distilled_logmel80.parquet"

cols = ['valence', 'arousal', 'soft_v', 'soft_a', 'confidence']
df = pd.read_parquet(PATH, columns=cols).dropna()

# 归一化 GT
df['gt_v'] = (df['valence'] - 1) / 4.0
df['gt_a'] = (df['arousal'] - 1) / 4.0

# ===============================
# 2. 基础指标
# ===============================
def metrics(y, y_hat):
    pcc, _ = pearsonr(y, y_hat)
    mae = np.mean(np.abs(y - y_hat))
    return pcc, mae

print("\n=== 📊 基础质量 ===")
print("Valence:", metrics(df['gt_v'], df['soft_v']))
print("Arousal:", metrics(df['gt_a'], df['soft_a']))

# ===============================
# 3. 偏置分析（关键）
# ===============================
df['v_error'] = df['soft_v'] - df['gt_v']
df['a_error'] = df['soft_a'] - df['gt_a']

print("\n=== 📉 偏置分析 ===")
print("Valence Bias (mean error):", df['v_error'].mean())
print("Arousal Bias (mean error):", df['a_error'].mean())

# ===============================
# 4. 区间一致性分析（分桶）
# ===============================
def bin_analysis(gt, pred, name):
    bins = np.linspace(0, 1, 6)
    df['bin'] = pd.cut(gt, bins)

    results = []
    for b, g in df.groupby('bin'):
        if len(g) < 50:
            continue
        pcc, mae = metrics(g[gt.name], g[pred.name])
        results.append((str(b), len(g), pcc, mae))

    print(f"\n=== 📊 {name} 分段表现 ===")
    for r in results:
        print(r)

bin_analysis(df['gt_v'], df['soft_v'], "Valence")
bin_analysis(df['gt_a'], df['soft_a'], "Arousal")

# ===============================
# 5. Confidence 真实性分析
# ===============================
df['conf_bin'] = pd.qcut(df['confidence'], 4, labels=False)

print("\n=== 📈 Confidence 可靠性 ===")
for i, g in df.groupby('conf_bin'):
    pcc_v, _ = metrics(g['gt_v'], g['soft_v'])
    pcc_a, _ = metrics(g['gt_a'], g['soft_a'])
    print(f"Level {i} | size={len(g)} | V={pcc_v:.3f} | A={pcc_a:.3f}")

# ===============================
# 6. 灾难性样本检测
# ===============================
df['total_error'] = np.abs(df['v_error']) + np.abs(df['a_error'])

bad = df.sort_values("total_error", ascending=False).head(20)

print("\n=== 💥 Top错误样本 ===")
print(bad[['gt_v','soft_v','gt_a','soft_a','confidence']])

# ===============================
# 7. 一致性评分（关键指标）
# ===============================
df['agreement'] = 1 - (np.abs(df['v_error']) + np.abs(df['a_error'])) / 2

print("\n=== 🤝 一致性 ===")
print("Mean agreement:", df['agreement'].mean())

# ===============================
# 8. 可视化（核心诊断）
# ===============================
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.scatterplot(x=df['gt_v'], y=df['soft_v'], alpha=0.3)
plt.title("Valence Alignment")

plt.subplot(1,3,2)
sns.scatterplot(x=df['gt_a'], y=df['soft_a'], alpha=0.3)
plt.title("Arousal Alignment")

plt.subplot(1,3,3)
sns.histplot(df['confidence'], bins=30)
plt.title("Confidence Distribution")

plt.tight_layout()
plt.savefig("audit_full.png", dpi=150)
plt.show()

print("\n✅ 分析完成：audit_full.png 已生成")