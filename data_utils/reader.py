import random
import sys
import warnings
from datetime import datetime
import joblib
from data_utils.utils import audio_features

warnings.filterwarnings("ignore")

import librosa
import numpy as np
from torch.utils import data

# 加载并预处理音频
def load_audio(audio_path, mode='train', sr=16000, chunk_duration=3, augmentors=None, use_noise_aug=True):
    wav, _ = librosa.load(audio_path, sr=sr)
    num_chunk_samples = int(chunk_duration * sr)

    # 短音频补零填充
    if len(wav) < num_chunk_samples:
        pad_len = num_chunk_samples - len(wav)
        wav = np.pad(wav, (0, pad_len), mode='constant')
    else:
        # 随机裁剪
        if mode == 'train':
            start = random.randint(0, len(wav) - num_chunk_samples)
        else:
            start = 0
        wav = wav[start:start + num_chunk_samples]

    # 随机静音增强（可选）
    if mode == 'train' and random.random() > 0.5:
        silence_len = random.randint(1, sr // 4)
        wav[:silence_len] = 0
        wav[-silence_len:] = 0

    # 数据增强（根据开关控制是否启用噪声相关增强）
    if mode == 'train' and augmentors:
        for key, augmenter in augmentors.items():
            # 如果关闭噪声增强，跳过 'combinedNoise' 及相关增强器
            if not use_noise_aug and key in ['combinedNoise', 'noise', 'volume', 'speed']:
                continue
            wav = augmenter(wav)

    return audio_features(wav, sr)


# 数据加载器，用于加载和预处理音频数据
class CustomDataset(data.Dataset):
    def __init__(self,
                 data_list_path,
                 scaler_path,
                 mode='train',
                 sr=16000,
                 chunk_duration=3,
                 augmentors=None,
                 use_noise_aug=True  # 新增：噪声增强开关
                 ):
        super().__init__()
        self.lines = []
        if data_list_path:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.augmentors = augmentors  # 增强器字典（需包含'combinedNoise'键）
        self.scaler = joblib.load(scaler_path)
        self.use_noise_aug = use_noise_aug  # 保存开关状态

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].strip().split('\t')
            # 传递噪声增强开关到load_audio函数
            features = load_audio(
                audio_path,
                self.mode,
                self.sr,
                self.chunk_duration,
                self.augmentors,
                use_noise_aug=self.use_noise_aug  # 控制是否启用噪声增强
            )
            features = self.scaler.transform([features]).squeeze().astype(np.float32)
            return features, np.int64(label)
        except Exception as e:
            print(f"[{datetime.now()}] 加载出错: {self.lines[idx]} -> {e}", file=sys.stderr)
            return self.__getitem__(random.randint(0, len(self.lines) - 1))

    def __len__(self):
        return len(self.lines)