import librosa
import numpy as np

def audio_features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))

    # 提取 pitch 信息（只取最大能量对应频率）
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = [pitches[:, i][magnitudes[:, i].argmax()] for i in range(pitches.shape[1])]
    pitch = np.array(pitch)
    pitch = pitch[pitch > 0]  # 去除无效值
    pitchmean = np.mean(pitch) if pitch.size else 0
    pitchstd = np.std(pitch) if pitch.size else 0
    pitchmax = np.max(pitch) if pitch.size else 0
    pitchmin = np.min(pitch) if pitch.size else 0
    pitch_tuning_offset = librosa.pitch_tuning(pitches) if pitch.size else 0

    # 谱特征
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))
    S, _ = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    rmse = librosa.feature.rms(S=S)[0]
    meanrms, stdrms, maxrms = np.mean(rmse), np.std(rmse), np.max(rmse)

    # MFCC 只提取一次
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T
    mfccs = np.mean(mfcc, axis=0)
    mfccsstd = np.std(mfcc, axis=0)
    mfccmax = np.max(mfcc, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    stats = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude,
        np.mean(cent), np.std(cent), np.max(cent), stdMagnitude,
        pitchmean, pitchmax, pitchstd, pitch_tuning_offset,
        meanrms, maxrms, stdrms
    ])

    return np.concatenate([stats, mfccs, mfccsstd, mfccmax, chroma, mel, contrast]).astype(np.float32)
