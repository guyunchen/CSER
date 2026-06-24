import os
import pandas as pd
import numpy as np


def inspect_parquet(folder_path="MESS"):
    # 待检查的文件列表
    files = ["train_v3.parquet", "test_v3.parquet"]

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"❌ 未找到文件: {file_path}，请确认路径是否正确。")
            continue

        print("=" * 75)
        print(f"📂 正在分析文件: {file_path}")
        print("=" * 75)

        try:
            # 读取 Parquet
            df = pd.read_parquet(file_path)

            # 1. 数据集基础大小
            num_rows, num_cols = df.shape
            print(f"✨ 数据规模: {num_rows} 行, {num_cols} 列")

            # 2. 字段名和数据类型
            print("\n📝 字段和数据类型 (Data Types):")
            for col, dtype in df.dtypes.items():
                print(f"   - {col}: {dtype}")

            # 3. 打印前 2 行样例（限制了显示宽度防止特征刷屏）
            print("\n🔍 数据样例 (前 2 行):")
            pd.set_option('display.max_colwidth', 50)  # 限制大特征向量的显示长度
            print(df.head(2).to_string())

            # 4. 检查特征维度 (如果是 logmel 特征等序列数据)
            feature_cols = [c for c in df.columns if 'mel' in c or 'feat' in c or 'audio' in c]
            for feat_col in feature_cols:
                first_item = df[feat_col].iloc[0]
                if isinstance(first_item, (list, np.ndarray)):
                    arr = np.array(first_item)
                    print(f"\n📊 特征维度分析: 列 '{feat_col}' 首行样本的形状 (Shape): {arr.shape}")

            # 5. 分析标签分布情况
            # 过滤掉特征列，剩下的基本是标签或ID列
            potential_labels = [col for col in df.columns if col not in feature_cols]
            if potential_labels:
                print("\n🎯 标签/属性分布分析:")
                for label_col in potential_labels:
                    unique_count = df[label_col].nunique()
                    if unique_count <= 25:  # 认为是分类标签/极性标签
                        print(f"   👉 列 [{label_col}] 类别分布 (Value Counts):")
                        print(df[label_col].value_counts().to_string())
                    else:  # 连续型标签（如 VAD）或者文件名/ID
                        if pd.api.types.is_numeric_dtype(df[label_col]):
                            print(f"   👉 列 [{label_col}] 数值分布统计:")
                            print(df[label_col].describe().to_string())
                        else:
                            print(f"   👉 列 [{label_col}] 独特值数量: {unique_count} 个（可能是文本或文件名ID）")

        except Exception as e:
            print(f"❌ 读取/解析该文件时发生错误: {e}")

        print("\n" + "=" * 75 + "\n")


if __name__ == "__main__":
    # 运行检查
    inspect_parquet("dataset/IEMOCAP/random_data")