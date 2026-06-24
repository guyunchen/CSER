import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple, Union
import warnings
from data_utils.normalization import normalize_logmel

warnings.filterwarnings('ignore')

# 假设这些模块已经按照之前的代码结构创建
try:
    from models.continuous_emotion_lnn import ContinuousEmotionLNN, RealTimeEmotionPredictor
    from utils.visualization import EmotionCurveVisualizer
except ImportError:
    print("Warning: Some modules not found. Creating simplified versions...")


    # 临时简化版本
    class ContinuousEmotionLNN(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()


    class RealTimeEmotionPredictor:
        def __init__(self, model):
            self.model = model


    class EmotionCurveVisualizer:
        def plot_emotion_curves(self, **kwargs):
            return plt.figure()


class ContinuousEmotionInferencer:
    """连续情绪推理器 - 完整版本"""

    def __init__(self, checkpoint_path: str, device: str = None,
                 feature_config: Dict = None):
        """
        初始化推理器

        Args:
            checkpoint_path: 模型检查点路径
            device: 计算设备 ('cuda' 或 'cpu')
            feature_config: 特征提取配置
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # 加载检查点
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})

        # 设置默认配置
        config.setdefault('input_dim', 312)
        config.setdefault('hidden_dim', 256)
        config.setdefault('num_liquid_layers', 2)
        config.setdefault('num_scales', 3)
        config.setdefault('use_multi_scale', True)
        config.setdefault('use_class_head', False)

        # 初始化模型
        self.model = ContinuousEmotionLNN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_liquid_layers=config.get('num_liquid_layers', 2),
            num_scales=config.get('num_scales', 3),
            dropout=0.0,  # 推理时关闭dropout
            use_multi_scale=config.get('use_multi_scale', True),
            use_class_head=config.get('use_class_head', False)
        ).to(self.device)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            # 处理可能的键不匹配
            model_state_dict = checkpoint['model_state_dict']
            current_state_dict = self.model.state_dict()

            # 检查键是否匹配
            mismatched_keys = [k for k in model_state_dict.keys()
                               if k not in current_state_dict]
            if mismatched_keys:
                print(f"Warning: {len(mismatched_keys)} keys mismatched. Trying to load compatible ones...")

                # 只加载匹配的键
                matched_state_dict = {}
                for k, v in model_state_dict.items():
                    if k in current_state_dict and v.shape == current_state_dict[k].shape:
                        matched_state_dict[k] = v

                self.model.load_state_dict(matched_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict)
        else:
            print("Warning: No model_state_dict found in checkpoint")

        self.model.eval()
        print(f"Model loaded successfully")

        # 初始化实时预测器
        self.realtime_predictor = RealTimeEmotionPredictor(self.model, cache_size=100)

        # 可视化工具
        self.visualizer = EmotionCurveVisualizer()

        # 音频特征提取参数
        if feature_config is None:
            feature_config = {}

        self.sr = feature_config.get('sr', 16000)
        self.n_mels = feature_config.get('n_mels', config['input_dim'])
        self.n_fft = feature_config.get('n_fft', 2048)
        self.hop_length = feature_config.get('hop_length', 160)
        self.win_length = feature_config.get('win_length', 400)
        self.fmin = feature_config.get('fmin', 50)
        self.fmax = feature_config.get('fmax', 8000)

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"inference_results/{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Inference device: {self.device}")
        print(f"Feature config: SR={self.sr}, Mels={self.n_mels}, Hop={self.hop_length}")
        print(f"Results will be saved to: {self.output_dir}")

    def extract_features(self, audio_path: Union[str, np.ndarray],
                         sr: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        从音频文件或数组提取LogMel特征

        Args:
            audio_path: 音频文件路径或音频数组
            sr: 采样率（如果提供音频数组）

        Returns:
            features: LogMel特征 (时间, 频率)
            audio: 原始音频
            sr: 采样率
        """
        # 加载音频
        if isinstance(audio_path, (str, Path)):
            audio, audio_sr = librosa.load(str(audio_path), sr=self.sr)
        else:
            # 已经是音频数组
            audio = audio_path
            audio_sr = sr if sr is not None else self.sr

            # 如果需要，重新采样
            if audio_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=self.sr)
                audio_sr = self.sr

        # 提取LogMel频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=audio_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # 转换为dB
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 转置为 (时间, 频率)
        log_mel = log_mel.T

        log_mel = normalize_logmel(log_mel)

        return log_mel, audio, audio_sr

    def preprocess_audio_chunk(self, audio_chunk: np.ndarray,
                               sr: int = None) -> torch.Tensor:
        """
        预处理音频片段用于实时预测

        Args:
            audio_chunk: 音频片段数组
            sr: 采样率

        Returns:
            预处理后的特征张量
        """
        if sr is not None and sr != self.sr:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.sr)

        # 提取特征
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max).T

        log_mel = normalize_logmel(log_mel)

        # 转换为张量
        features_tensor = torch.FloatTensor(log_mel).unsqueeze(0).to(self.device)

        return features_tensor

    def predict_audio_file(self, audio_path: Union[str, np.ndarray],
                           visualize: bool = True, save_results: bool = True,
                           return_details: bool = False) -> Dict:
        """
        预测整个音频文件的情绪曲线

        Args:
            audio_path: 音频文件路径或音频数组
            visualize: 是否生成可视化图表
            save_results: 是否保存结果
            return_details: 是否返回详细输出

        Returns:
            包含预测结果的字典
        """
        print(f"\n{'=' * 60}")
        print(f"Processing: {audio_path if isinstance(audio_path, str) else 'Audio array'}")
        print(f"{'=' * 60}")

        # 提取特征
        start_time = datetime.now()
        features, audio, sr = self.extract_features(audio_path)
        feature_time = (datetime.now() - start_time).total_seconds()

        print(f"Audio duration: {len(audio) / sr:.2f}s")
        print(f"Features shape: {features.shape}")
        print(f"Feature extraction time: {feature_time:.2f}s")

        # 转换为张量
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # 前向传播
        start_time = datetime.now()
        with torch.no_grad():
            outputs = self.model(features_tensor, return_attention=True, return_scales=True)
        inference_time = (datetime.now() - start_time).total_seconds()

        print(f"Inference time: {inference_time:.2f}s")
        print(f"Real-time factor: {(len(audio) / sr) / inference_time:.2f}x")

        # 获取预测结果
        predictions = outputs['continuous'][0].cpu().numpy()  # (seq_len, 2)

        # 计算时间戳
        time_stamps = np.arange(len(predictions)) * self.hop_length / sr

        # 计算统计信息
        stats = self._calculate_statistics(predictions, time_stamps)

        # 保存结果
        if save_results:
            save_dir = self._save_results(audio_path, predictions, time_stamps,
                                          outputs, features, audio, sr, stats)

            if visualize:
                self._visualize_results(audio_path, predictions, time_stamps,
                                        outputs, features, audio, sr, save_dir, stats)

        # 准备返回结果
        result = {
            'predictions': predictions,
            'time_stamps': time_stamps,
            'statistics': stats,
            'audio_info': {
                'duration': len(audio) / sr,
                'sampling_rate': sr,
                'num_frames': len(predictions)
            },
            'timing': {
                'feature_extraction': feature_time,
                'inference': inference_time
            }
        }

        if return_details:
            result.update({
                'attention': outputs.get('attention'),
                'tau_estimations': outputs.get('tau_estimations'),
                'features': features,
                'audio': audio if isinstance(audio_path, str) else None,
                'model_outputs': outputs
            })

        return result

    def predict_realtime_stream(self, audio_chunk: np.ndarray,
                                sr: int = None, reset_cache: bool = False) -> Dict:
        """
        实时流式预测

        Args:
            audio_chunk: 音频片段数组
            sr: 采样率
            reset_cache: 是否重置缓存

        Returns:
            实时预测结果
        """
        if reset_cache:
            self.realtime_predictor.reset_cache()
            print("Cache reset")

        # 预处理音频
        features_tensor = self.preprocess_audio_chunk(audio_chunk, sr)

        # 实时预测
        with torch.no_grad():
            outputs = self.realtime_predictor(features_tensor)

        # 获取最新预测
        latest_predictions = outputs['continuous'].cpu().numpy()

        # 计算时间戳
        chunk_duration = len(audio_chunk) / (sr if sr is not None else self.sr)
        if hasattr(self, 'realtime_start_time'):
            self.realtime_start_time += chunk_duration
        else:
            self.realtime_start_time = 0

        time_stamps = np.linspace(self.realtime_start_time - chunk_duration,
                                  self.realtime_start_time, len(latest_predictions))

        return {
            'continuous': latest_predictions,
            'time_stamps': time_stamps,
            'full_context': outputs.get('full_context'),
            'attention': outputs.get('attention'),
            'current_time': self.realtime_start_time
        }

    def predict_folder(self, folder_path: str, extension: str = '.wav',
                       save_results: bool = True) -> List[Dict]:
        """
        批量预测文件夹中的音频文件

        Args:
            folder_path: 文件夹路径
            extension: 音频文件扩展名
            save_results: 是否保存结果

        Returns:
            所有文件的预测结果列表
        """
        folder_path = Path(folder_path)
        audio_files = list(folder_path.glob(f"*{extension}"))

        if not audio_files:
            print(f"No {extension} files found in {folder_path}")
            return []

        print(f"\n{'=' * 60}")
        print(f"Batch processing {len(audio_files)} audio files")
        print(f"{'=' * 60}")

        all_results = []

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

            try:
                result = self.predict_audio_file(
                    str(audio_file),
                    visualize=False,  # 批量处理时不单独可视化
                    save_results=save_results,
                    return_details=False
                )
                all_results.append(result)

            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                continue

        # 保存批量处理摘要
        if save_results and all_results:
            self._save_batch_summary(all_results, folder_path)

        return all_results

    def _calculate_statistics(self, predictions: np.ndarray,
                              time_stamps: np.ndarray) -> Dict:
        """计算预测结果的统计信息"""
        valence = predictions[:, 0]
        arousal = predictions[:, 1]

        stats = {
            'valence': {
                'mean': float(np.mean(valence)),
                'std': float(np.std(valence)),
                'min': float(np.min(valence)),
                'max': float(np.max(valence)),
                'median': float(np.median(valence)),
                'q1': float(np.percentile(valence, 25)),
                'q3': float(np.percentile(valence, 75)),
                'positive_ratio': float(np.sum(valence > 0) / len(valence)),
                'negative_ratio': float(np.sum(valence < 0) / len(valence)),
            },
            'arousal': {
                'mean': float(np.mean(arousal)),
                'std': float(np.std(arousal)),
                'min': float(np.min(arousal)),
                'max': float(np.max(arousal)),
                'median': float(np.median(arousal)),
                'q1': float(np.percentile(arousal, 25)),
                'q3': float(np.percentile(arousal, 75)),
                'high_ratio': float(np.sum(arousal > 0.5) / len(arousal)),
                'low_ratio': float(np.sum(arousal < -0.5) / len(arousal)),
            },
            'combined': {
                'mean_intensity': float(np.mean(np.abs(predictions))),
                'max_intensity': float(np.max(np.abs(predictions))),
                'variability': float(np.mean(np.std(predictions, axis=0))),
                'dominant_quadrant': self._get_dominant_quadrant(valence, arousal),
                'emotional_entropy': self._calculate_entropy(predictions),
            }
        }

        return stats

    def _get_dominant_quadrant(self, valence: np.ndarray, arousal: np.ndarray) -> str:
        """获取主导情感象限"""
        quadrants = {
            'Q1': np.sum((valence > 0) & (arousal > 0)),  # 高兴
            'Q2': np.sum((valence < 0) & (arousal > 0)),  # 愤怒/紧张
            'Q3': np.sum((valence < 0) & (arousal < 0)),  # 悲伤
            'Q4': np.sum((valence > 0) & (arousal < 0)),  # 平静/放松
        }

        dominant = max(quadrants.items(), key=lambda x: x[1])
        quadrant_names = {
            'Q1': 'Positive High-Arousal (Happy/Excited)',
            'Q2': 'Negative High-Arousal (Angry/Stressed)',
            'Q3': 'Negative Low-Arousal (Sad/Depressed)',
            'Q4': 'Positive Low-Arousal (Calm/Relaxed)'
        }

        return f"{dominant[0]} - {quadrant_names[dominant[0]]} ({dominant[1] / len(valence) * 100:.1f}%)"

    def _calculate_entropy(self, predictions: np.ndarray, num_bins: int = 10) -> float:
        """计算情绪分布的熵"""
        # 将2D情绪空间分箱
        valence_bins = np.linspace(-1, 1, num_bins)
        arousal_bins = np.linspace(-1, 1, num_bins)

        # 创建2D直方图
        hist, _, _ = np.histogram2d(predictions[:, 0], predictions[:, 1],
                                    bins=[valence_bins, arousal_bins])
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # 移除空箱
        hist_flat = hist_flat / hist_flat.sum()  # 归一化

        # 计算熵
        entropy = -np.sum(hist_flat * np.log2(hist_flat + 1e-10))

        return float(entropy)

    def _save_results(self, audio_path: Union[str, np.ndarray],
                      predictions: np.ndarray, time_stamps: np.ndarray,
                      outputs: Dict, features: np.ndarray, audio: np.ndarray,
                      sr: int, stats: Dict) -> Path:
        """保存预测结果"""
        if isinstance(audio_path, str):
            audio_name = Path(audio_path).stem
        else:
            audio_name = f"audio_{datetime.now().strftime('%H%M%S')}"

        # 为当前音频创建子目录
        save_dir = self.output_dir / audio_name
        save_dir.mkdir(exist_ok=True)

        # 保存预测数据
        np.save(save_dir / 'predictions.npy', predictions)
        np.save(save_dir / 'time_stamps.npy', time_stamps)
        np.save(save_dir / 'features.npy', features)

        # 保存音频（如果不是太大）
        if isinstance(audio_path, str) and len(audio) / sr < 60:  # 小于60秒
            sf.write(save_dir / 'audio.wav', audio, sr)

        # 保存统计信息
        with open(save_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

        # 保存注意力权重
        if outputs.get('attention') is not None:
            attention = outputs['attention'][0].cpu().numpy()
            np.save(save_dir / 'attention.npy', attention)

        # 保存τ估计
        if outputs.get('tau_estimations') is not None:
            tau_list = [t[0].cpu().numpy() for t in outputs['tau_estimations']]
            np.save(save_dir / 'tau_estimations.npy', tau_list)

        print(f"Results saved to: {save_dir}")

        return save_dir

    def _visualize_results(self, audio_path: Union[str, np.ndarray],
                           predictions: np.ndarray, time_stamps: np.ndarray,
                           outputs: Dict, features: np.ndarray, audio: np.ndarray,
                           sr: int, save_dir: Path, stats: Dict):
        """可视化预测结果"""
        audio_name = Path(audio_path).stem if isinstance(audio_path, str) else "audio"

        # 1. 情绪曲线图
        fig1 = self.visualizer.plot_emotion_curves(
            predictions=predictions,
            timestamps=time_stamps,
            title=f'Emotion Dynamics - {audio_name}'
        )
        fig1.savefig(save_dir / 'emotion_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # 2. 注意力权重图
        if outputs.get('attention') is not None:
            attention = outputs['attention'][0].cpu().numpy()
            fig2 = self.visualizer.plot_attention_weights(
                attention_weights=attention,
                timestamps=time_stamps,
                title=f'Attention Weights - {audio_name}'
            )
            fig2.savefig(save_dir / 'attention_weights.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)

        # 3. τ动态图
        if outputs.get('tau_estimations') is not None:
            tau_list = [t[0].cpu().numpy() for t in outputs['tau_estimations']]
            fig3 = self.visualizer.plot_tau_dynamics(
                tau_estimations=tau_list,
                timestamps=time_stamps[:len(tau_list[0])] if tau_list else None,
                title=f'Tau Dynamics - {audio_name}'
            )
            if fig3 is not None:
                fig3.savefig(save_dir / 'tau_dynamics.png', dpi=150, bbox_inches='tight')
                plt.close(fig3)

        # 4. 频谱图
        fig4, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 波形
        axes[0].plot(np.arange(len(audio)) / sr, audio, 'b-', alpha=0.7)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Audio Waveform - {audio_name}')
        axes[0].grid(True, alpha=0.3)

        # 频谱图
        axes[1].imshow(features.T, aspect='auto', origin='lower',
                       extent=[time_stamps[0], time_stamps[-1], 0, self.n_mels],
                       cmap='hot')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Mel Band')
        axes[1].set_title('Log-Mel Spectrogram')
        axes[1].set_ylim([0, 50])  # 只显示前50个mel band

        plt.tight_layout()
        plt.savefig(save_dir / 'spectrogram.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)

        # 5. 统计信息图
        fig5 = plt.figure(figsize=(10, 6))

        # 创建统计信息文本
        stats_text = f"Audio: {audio_name}\n"
        stats_text += f"Duration: {len(audio) / sr:.2f}s\n"
        stats_text += f"Frames: {len(predictions)}\n\n"

        stats_text += "Valence Statistics:\n"
        for k, v in stats['valence'].items():
            if isinstance(v, float):
                stats_text += f"  {k}: {v:.3f}\n"

        stats_text += "\nArousal Statistics:\n"
        for k, v in stats['arousal'].items():
            if isinstance(v, float):
                stats_text += f"  {k}: {v:.3f}\n"

        stats_text += f"\nDominant Quadrant:\n  {stats['combined']['dominant_quadrant']}\n"
        stats_text += f"Emotional Entropy: {stats['combined']['emotional_entropy']:.3f}"

        plt.text(0.1, 0.5, stats_text, fontsize=10,
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / 'statistics_summary.png', dpi=150, bbox_inches='tight')
        plt.close(fig5)

        print(f"Visualizations saved to {save_dir}")

    def _save_batch_summary(self, all_results: List[Dict], folder_path: Path):
        """保存批量处理摘要"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'folder': str(folder_path),
            'num_files': len(all_results),
            'results': []
        }

        for i, result in enumerate(all_results):
            file_summary = {
                'index': i + 1,
                'valence_mean': result['statistics']['valence']['mean'],
                'arousal_mean': result['statistics']['arousal']['mean'],
                'dominant_quadrant': result['statistics']['combined']['dominant_quadrant'],
                'duration': result['audio_info']['duration'],
                'num_frames': result['audio_info']['num_frames']
            }
            summary['results'].append(file_summary)

        # 计算总体统计
        valence_means = [r['statistics']['valence']['mean'] for r in all_results]
        arousal_means = [r['statistics']['arousal']['mean'] for r in all_results]

        summary['overall_stats'] = {
            'mean_valence': float(np.mean(valence_means)),
            'std_valence': float(np.std(valence_means)),
            'mean_arousal': float(np.mean(arousal_means)),
            'std_arousal': float(np.std(arousal_means)),
            'total_duration': sum(r['audio_info']['duration'] for r in all_results)
        }

        # 保存摘要
        summary_path = self.output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nBatch summary saved to: {summary_path}")

        # 创建批量处理报告
        self._create_batch_report(summary, all_results)

    def _create_batch_report(self, summary: Dict, all_results: List[Dict]):
        """创建批量处理报告"""
        report_path = self.output_dir / 'batch_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BATCH PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Processing Time: {summary['timestamp']}\n")
            f.write(f"Folder: {summary['folder']}\n")
            f.write(f"Number of Files: {summary['num_files']}\n")
            f.write(f"Total Duration: {summary['overall_stats']['total_duration']:.2f}s\n\n")

            f.write("=" * 60 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Mean Valence: {summary['overall_stats']['mean_valence']:.3f} "
                    f"(±{summary['overall_stats']['std_valence']:.3f})\n")
            f.write(f"Mean Arousal: {summary['overall_stats']['mean_arousal']:.3f} "
                    f"(±{summary['overall_stats']['std_arousal']:.3f})\n\n")

            f.write("=" * 60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 60 + "\n\n")

            for i, result in enumerate(all_results, 1):
                stats = result['statistics']
                f.write(f"File {i}:\n")
                f.write(f"  Duration: {result['audio_info']['duration']:.2f}s\n")
                f.write(f"  Valence: {stats['valence']['mean']:.3f} "
                        f"(min: {stats['valence']['min']:.3f}, "
                        f"max: {stats['valence']['max']:.3f})\n")
                f.write(f"  Arousal: {stats['arousal']['mean']:.3f} "
                        f"(min: {stats['arousal']['min']:.3f}, "
                        f"max: {stats['arousal']['max']:.3f})\n")
                f.write(f"  Dominant: {stats['combined']['dominant_quadrant']}\n")
                f.write("\n")

        print(f"Batch report saved to: {report_path}")

    def export_to_csv(self, results: Dict, output_path: str = None):
        """将结果导出为CSV格式"""
        import pandas as pd

        if output_path is None:
            output_path = self.output_dir / 'predictions.csv'

        # 创建DataFrame
        df = pd.DataFrame({
            'time': results['time_stamps'],
            'valence': results['predictions'][:, 0],
            'arousal': results['predictions'][:, 1]
        })

        # 添加衍生特征
        df['intensity'] = np.sqrt(df['valence'] ** 2 + df['arousal'] ** 2)
        df['angle'] = np.arctan2(df['arousal'], df['valence'])

        # 保存为CSV
        df.to_csv(output_path, index=False)
        print(f"Predictions exported to CSV: {output_path}")

        return df


# 使用示例
if __name__ == "__main__":
    # 示例配置
    CHECKPOINT_PATH = "checkpoints/best_model.pth"  # 修改为你的模型路径

    # 创建推理器
    inferencer = ContinuousEmotionInferencer(
        checkpoint_path=CHECKPOINT_PATH,
        device='cuda',  # 或 'cpu'
        feature_config={
            'sr': 16000,
            'n_mels': 312,
            'hop_length': 160,
            'n_fft': 2048
        }
    )

    # 示例1: 预测单个音频文件
    audio_file = "test_audio.wav"  # 修改为你的音频文件路径
    if os.path.exists(audio_file):
        result = inferencer.predict_audio_file(
            audio_file,
            visualize=True,
            save_results=True
        )

        # 导出为CSV
        inferencer.export_to_csv(result)
    else:
        print(f"Audio file not found: {audio_file}")

    # 示例2: 批量预测文件夹
    audio_folder = "audio_samples"  # 修改为你的音频文件夹路径
    if os.path.exists(audio_folder) and os.path.isdir(audio_folder):
        batch_results = inferencer.predict_folder(
            audio_folder,
            extension='.wav',
            save_results=True
        )

    # 示例3: 实时流式预测
    # 这需要一个音频流输入，例如从麦克风
    print("\nReal-time inference example:")
    print("To use real-time prediction, provide audio chunks from a stream")

    # 示例音频块
    test_chunk = np.random.randn(16000)  # 1秒音频，16kHz

    realtime_result = inferencer.predict_realtime_stream(
        test_chunk,
        sr=16000,
        reset_cache=True
    )

    if realtime_result['continuous'] is not None:
        print(f"Real-time prediction shape: {realtime_result['continuous'].shape}")
