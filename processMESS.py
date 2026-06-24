import pandas as pd
import numpy as np
import librosa
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class MESSProcessor:
    def __init__(self, sr=16000, n_mels=80):
        self.sr = sr
        self.n_mels = n_mels

    def extract_logmel(self, path):
        try:
            # 加载音频
            y, _ = librosa.load(path, sr=self.sr)
            # 预加重
            y = librosa.effects.preemphasis(y)
            # 提取 Mel 频谱 (25ms 窗长, 10ms 帧移)
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels,
                n_fft=400, hop_length=160
            )
            # 转为 Log 分贝
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # 转置为 (Time, 80)
            return log_mel.T.astype(np.float32).tolist()
        except Exception as e:
            print(f"提取失败 {path}: {e}")
            return None


def main():
    # 路径配置
    excel_path = r"E:\五邑大学\论文\情绪识别\音频情绪识别\数据集\MESS\Coded stimuli - final MESS.xlsx"
    audio_dir = r"E:\五邑大学\论文\情绪识别\音频情绪识别\数据集\MESS\MESS_Origin"
    output_dir = "dataset/MESS"
    os.makedirs(output_dir, exist_ok=True)

    processor = MESSProcessor()

    # 加载并解析 Excel
    print(f"正在读取表格: {excel_path}")
    df_meta = pd.read_excel(excel_path)
    data_list = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        file_code = str(row['code']).strip()
        # 修复点：匹配实际文件名后缀 _SCR.wav
        wav_path = os.path.join(audio_dir, f"{file_code}_SCR.wav")

        if os.path.exists(wav_path):
            feat = processor.extract_logmel(wav_path)
            if feat is not None:
                v = float(row['Valence']) / 100.0
                a = float(row['Arousal']) / 100.0

                data_list.append({
                    'logmel_80': feat,
                    'valence': v,
                    'arousal': a,
                    'file': file_code
                })

    full_df = pd.DataFrame(data_list)
    if full_df.empty:
        print("❌ 错误：未找到音频文件，请检查 audio_dir 路径是否正确。")
        return

    # 划分训练集和测试集 (9:1)
    train_df, test_df = train_test_split(full_df, test_size=0.1, random_state=42)

    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    print(f"\n✅ 处理完成！")
    print(f"总样本数: {len(full_df)}")
    print(f"训练集: {len(train_df)} | 测试集: {len(test_df)}")
    print(f"保存路径: {output_dir}")


if __name__ == "__main__":
    main()