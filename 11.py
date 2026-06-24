import pickle
import numpy as np

# 👉 改成你自己的路径
PKL_PATH = r"E:\五邑大学\论文\情绪识别\音频情绪识别\数据集\MOSI\aligned_50.pkl"

def inspect_mosi_labels(pkl_path):
    print("=" * 60)
    print("🔍 MOSI 数据集 标签深度探测")
    print("=" * 60)

    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 查看子集
    splits = ['train', 'valid', 'test']
    for split in splits:
        print(f"\n📂 子集: {split}")
        subset = data[split]

        # 打印这个子集里所有 9 个键（就是你看到的9个字段）
        print(f"  包含字段: {list(subset.keys())}")

        # 取出第一个样本，查看所有内容
        sample = {k: v[0] for k, v in subset.items()}
        print(f"  第一个样本的所有值：")
        for k, v in sample.items():
            # 处理数组/数值
            if isinstance(v, (np.ndarray, list)):
                v = np.array(v)
                print(f"    {k:<15} → shape={v.shape}, value={v}")
            else:
                print(f"    {k:<15} → {v}")

        print("-" * 50)

    # ==================== 统计标签 ====================
    print("\n" + "="*60)
    print("📊 MOSI 数据集标签统计（所有子集）")
    print("="*60)

    for split in splits:
        subset = data[split]
        print(f"\n【{split} 集】")
        for key in subset.keys():
            values = subset[key]
            try:
                arr = np.array(values)
                print(f"  {key:<15} → 数量={len(arr)}, 范围=[{arr.min():.3f}, {arr.max():.3f}], 均值={arr.mean():.3f}")
            except:
                print(f"  {key:<15} → 非数值标签，数量={len(values)}")

if __name__ == '__main__':
    inspect_mosi_labels(PKL_PATH)