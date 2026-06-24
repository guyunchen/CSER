import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 设置中文字体（避免可视化乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 核心配置 =====================
raw_data_dir = r"dataset/IEMOCAP"
new_data_dir = r"dataset/IEMOCAP/random_data"
os.makedirs(new_data_dir, exist_ok=True)

# ===================== 1. 读取并整合分片 =====================
parquet_files = sorted(glob.glob(os.path.join(raw_data_dir, "train-*.parquet")))
df_list = [pd.read_parquet(f, engine="pyarrow") for f in parquet_files]
df_full = pd.concat(df_list, ignore_index=True)

# ===================== 2. VAD字段规范化 =====================
df_full["valence"] = df_full["EmoVal"]
df_full["arousal"] = df_full["EmoAct"]
df_full["dominance"] = df_full["EmoDom"]

# ===================== 3. 新增情绪极性 =====================
def map_to_polarity(major_emotion):
    positive = ["happy", "excited"]
    negative = ["angry", "sad", "frustrated", "fear", "disgust"]
    neutral = ["neutral", "surprise", "other"]
    if major_emotion in positive:
        return "Positive"
    elif major_emotion in negative:
        return "Negative"
    else:
        return "Neutral"

df_full["polarity"] = df_full["major_emotion"].apply(map_to_polarity)

# ===================== 4. 8:2分层划分（核心） =====================
df_train, df_test = train_test_split(
    df_full,
    test_size=0.2,
    random_state=42,
    stratify=df_full["polarity"]
)

# ===================== 5. 全面验证分布均匀性 =====================
print("="*50)
print("📊 分布均匀性验证报告")
print("="*50)

# 5.1 情绪极性分布验证
print("\n【1. 情绪极性分布（%）】")
dist_full = df_full["polarity"].value_counts(normalize=True) * 100
dist_train = df_train["polarity"].value_counts(normalize=True) * 100
dist_test = df_test["polarity"].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    "原数据集": dist_full,
    "训练集": dist_train,
    "测试集": dist_test
}).round(2)
print(dist_df)

# 5.2 核心情绪类别分布验证（major_emotion前5类）
print("\n【2. 核心情绪类别分布（%）】")
top5_emotions = df_full["major_emotion"].value_counts().head(5).index
dist_full_major = df_full[df_full["major_emotion"].isin(top5_emotions)]["major_emotion"].value_counts(normalize=True) * 100
dist_train_major = df_train[df_train["major_emotion"].isin(top5_emotions)]["major_emotion"].value_counts(normalize=True) * 100
dist_test_major = df_test[df_test["major_emotion"].isin(top5_emotions)]["major_emotion"].value_counts(normalize=True) * 100

dist_major_df = pd.DataFrame({
    "原数据集": dist_full_major,
    "训练集": dist_train_major,
    "测试集": dist_test_major
}).round(2)
print(dist_major_df)

# 5.3 VAD分布验证（均值对比）
print("\n【3. VAD均值对比】")
vad_cols = ["valence", "arousal", "dominance"]
vad_full = df_full[vad_cols].mean().round(3)
vad_train = df_train[vad_cols].mean().round(3)
vad_test = df_test[vad_cols].mean().round(3)

vad_df = pd.DataFrame({
    "原数据集": vad_full,
    "训练集": vad_train,
    "测试集": vad_test
})
print(vad_df)

# ===================== 6. 可视化分布（直观确认） =====================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 6.1 极性分布饼图
axes[0].set_title("情绪极性分布对比", fontsize=12)
dist_train.plot.pie(ax=axes[0], label="训练集", autopct="%1.1f%%", startangle=90, ylabel="")
axes[1].set_title("测试集极性分布", fontsize=12)
dist_test.plot.pie(ax=axes[1], label="测试集", autopct="%1.1f%%", startangle=90, ylabel="")

plt.tight_layout()
plt.savefig(os.path.join(new_data_dir, "分布验证.png"), dpi=300, bbox_inches="tight")
print(f"\n✅ 分布可视化图已保存：{os.path.join(new_data_dir, '分布验证.png')}")

# ===================== 7. 保存最终数据集 =====================
df_train.to_parquet(os.path.join(new_data_dir, "train.parquet"), engine="pyarrow", compression="snappy")
df_test.to_parquet(os.path.join(new_data_dir, "test.parquet"), engine="pyarrow", compression="snappy")

print("\n" + "="*50)
print("✅ 数据集划分完成！")
print(f"📁 训练集：{new_data_dir}/train.parquet（{len(df_train)} 条）")
print(f"📁 测试集：{new_data_dir}/test.parquet（{len(df_test)} 条）")
print("✅ 训练/测试集分布均匀，可直接用于模型训练！")
print("="*50)