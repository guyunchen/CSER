import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 用于无头环境
import torch
from typing import Dict, List, Optional
import io
from PIL import Image


class EmotionCurveVisualizer:
    """情绪曲线可视化工具"""

    def __init__(self, figsize=(12, 8), dpi=100):
        self.figsize = figsize
        self.dpi = dpi

    def plot_emotion_curves(self, predictions, targets=None, timestamps=None,
                            title="Emotion Dynamics", save_path=None):
        """
        绘制情绪变化曲线

        Args:
            predictions: 预测值 (seq_len, 2) 或 (batch, seq_len, 2)
            targets: 真实值，同predictions形状
            timestamps: 时间戳
            title: 图表标题
            save_path: 保存路径

        Returns:
            matplotlib图像
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # 处理批量数据
        if predictions.ndim == 3:
            predictions = predictions[0]  # 取第一个样本

        if targets is not None and targets.ndim == 3:
            targets = targets[0]

        seq_len = predictions.shape[0]
        if timestamps is None:
            timestamps = np.arange(seq_len) / 100  # 假设100Hz

        fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi,
                                 sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # 情绪维度名称
        dim_names = ['Valence (Pleasure)', 'Arousal (Activation)']
        colors = ['blue', 'red']

        # 绘制主要情绪曲线
        ax1 = axes[0]
        for i in range(2):
            ax1.plot(timestamps, predictions[:, i],
                     label=f'Predicted {dim_names[i]}',
                     color=colors[i], linewidth=2, alpha=0.8)

            if targets is not None:
                ax1.plot(timestamps, targets[:, i],
                         label=f'Ground Truth {dim_names[i]}',
                         color=colors[i], linestyle='--', linewidth=1.5, alpha=0.6)

        ax1.set_ylabel('Emotion Value')
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-1.1, 1.1])

        # 绘制情绪空间轨迹
        ax2 = axes[1]
        scatter = ax2.scatter(predictions[:, 0], predictions[:, 1],
                              c=timestamps, cmap='viridis', alpha=0.6,
                              s=20, edgecolors='k', linewidth=0.5)

        # 添加轨迹线
        ax2.plot(predictions[:, 0], predictions[:, 1], 'k-', alpha=0.3, linewidth=0.5)

        # 标记起点和终点
        ax2.scatter(predictions[0, 0], predictions[0, 1],
                    c='green', s=100, marker='o', label='Start', edgecolors='k')
        ax2.scatter(predictions[-1, 0], predictions[-1, 1],
                    c='red', s=100, marker='s', label='End', edgecolors='k')

        ax2.set_xlabel('Valence')
        ax2.set_ylabel('Arousal')
        ax2.set_title('Emotion Space Trajectory')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_aspect('equal', 'box')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Time')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        return fig

    def plot_attention_weights(self, attention_weights, features=None,
                               timestamps=None, save_path=None):
        """
        绘制注意力权重热力图

        Args:
            attention_weights: 注意力权重 (seq_len,) 或 (num_heads, seq_len)
            features: 特征序列，用于相关性分析
            timestamps: 时间戳
            save_path: 保存路径
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        if attention_weights.ndim == 1:
            attention_weights = attention_weights.reshape(1, -1)

        num_heads, seq_len = attention_weights.shape
        if timestamps is None:
            timestamps = np.arange(seq_len) / 100

        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                 gridspec_kw={'height_ratios': [3, 1]})

        # 绘制注意力热力图
        ax1 = axes[0]
        im = ax1.imshow(attention_weights, aspect='auto',
                        cmap='YlOrRd', interpolation='nearest',
                        extent=[timestamps[0], timestamps[-1], 0, num_heads])

        ax1.set_yticks(np.arange(num_heads))
        ax1.set_yticklabels([f'Head {i}' for i in range(num_heads)])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Attention Head')
        ax1.set_title('Temporal Attention Weights')

        plt.colorbar(im, ax=ax1, label='Attention Weight')

        # 绘制注意力权重随时间的变化
        ax2 = axes[1]
        for i in range(num_heads):
            ax2.plot(timestamps, attention_weights[i],
                     label=f'Head {i}', alpha=0.7, linewidth=1.5)

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Weight')
        ax2.set_title('Attention Weight Dynamics')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        return fig

    def plot_tau_dynamics(self, tau_estimations, timestamps=None, save_path=None):
        """
        绘制时间常数τ的动态变化

        Args:
            tau_estimations: τ估计值列表，每个元素为(batch, seq_len)或(seq_len,)
            timestamps: 时间戳
            save_path: 保存路径
        """
        if not tau_estimations:
            return None

        fig, axes = plt.subplots(len(tau_estimations), 1,
                                 figsize=(12, 3 * len(tau_estimations)))

        if len(tau_estimations) == 1:
            axes = [axes]

        for idx, tau in enumerate(tau_estimations):
            if isinstance(tau, torch.Tensor):
                tau = tau.detach().cpu().numpy()

            if tau.ndim == 2:
                tau = tau[0]  # 取第一个样本

            seq_len = len(tau)
            if timestamps is None:
                timestamps = np.arange(seq_len) / 100

            ax = axes[idx]
            ax.plot(timestamps, tau, 'b-', linewidth=2, alpha=0.8)
            ax.fill_between(timestamps, 0, tau, alpha=0.3, color='blue')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'τ (Layer {idx})')
            ax.set_title(f'Dynamic Time Constant - Layer {idx}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)

        return fig

    def create_real_time_dashboard(self, predictions_history, attention_history=None,
                                   tau_history=None, window_size=50):
        """
        创建实时仪表板

        Args:
            predictions_history: 预测历史列表
            attention_history: 注意力历史
            tau_history: τ历史
            window_size: 显示窗口大小

        Returns:
            仪表板图像
        """
        fig = plt.figure(figsize=(15, 10))

        # 创建子图布局
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. 实时情绪曲线
        ax1 = fig.add_subplot(gs[0, :])
        if predictions_history:
            recent_preds = predictions_history[-window_size:]
            if isinstance(recent_preds, torch.Tensor):
                recent_preds = recent_preds.detach().cpu().numpy()

            time_points = np.arange(len(recent_preds))
            if recent_preds.ndim == 3:
                recent_preds = recent_preds[0]  # 取第一个样本

            ax1.plot(time_points, recent_preds[:, 0], 'b-',
                     label='Valence', linewidth=2)
            ax1.plot(time_points, recent_preds[:, 1], 'r-',
                     label='Arousal', linewidth=2)

            ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax1.fill_between(time_points, recent_preds[:, 0], 0,
                             alpha=0.2, color='blue')
            ax1.fill_between(time_points, recent_preds[:, 1], 0,
                             alpha=0.2, color='red')

        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Emotion Value')
        ax1.set_title('Real-time Emotion Dynamics')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-1.1, 1.1])

        # 2. 情绪空间轨迹
        ax2 = fig.add_subplot(gs[1, 0])
        if predictions_history and len(predictions_history) > 1:
            all_preds = np.vstack([p[0] if p.ndim == 3 else p
                                   for p in predictions_history[-window_size:]])

            scatter = ax2.scatter(all_preds[:, 0], all_preds[:, 1],
                                  c=np.arange(len(all_preds)),
                                  cmap='viridis', alpha=0.6, s=30)

            if len(all_preds) > 1:
                ax2.plot(all_preds[:, 0], all_preds[:, 1], 'k-',
                         alpha=0.3, linewidth=0.5)

            ax2.scatter(all_preds[0, 0], all_preds[0, 1],
                        c='green', s=100, marker='o', label='Start')
            ax2.scatter(all_preds[-1, 0], all_preds[-1, 1],
                        c='red', s=100, marker='s', label='End')

            plt.colorbar(scatter, ax=ax2, label='Time Step')

        ax2.set_xlabel('Valence')
        ax2.set_ylabel('Arousal')
        ax2.set_title('Emotion Space Trajectory')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_aspect('equal', 'box')

        # 3. 注意力权重
        ax3 = fig.add_subplot(gs[1, 1])
        if attention_history and len(attention_history) > 0:
            recent_attention = attention_history[-window_size:]
            if isinstance(recent_attention, torch.Tensor):
                recent_attention = recent_attention.detach().cpu().numpy()

            if recent_attention.ndim == 1:
                recent_attention = recent_attention.reshape(1, -1)

            im = ax3.imshow(recent_attention, aspect='auto',
                            cmap='YlOrRd', interpolation='nearest')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylabel('Attention Heads')
            ax3.set_title('Attention Weights')
            plt.colorbar(im, ax=ax3, label='Weight')

        # 4. 统计信息
        ax4 = fig.add_subplot(gs[2, :])
        if predictions_history:
            all_preds = np.vstack([p[0] if p.ndim == 3 else p
                                   for p in predictions_history[-window_size:]])

            stats = {
                'Mean Valence': np.mean(all_preds[:, 0]),
                'Mean Arousal': np.mean(all_preds[:, 1]),
                'Std Valence': np.std(all_preds[:, 0]),
                'Std Arousal': np.std(all_preds[:, 1]),
                'Max Valence': np.max(all_preds[:, 0]),
                'Min Valence': np.min(all_preds[:, 1]),
                'Dominant': 'Positive' if np.mean(all_preds[:, 0]) > 0 else 'Negative',
                'Intensity': 'High' if np.mean(np.abs(all_preds[:, 1])) > 0.5 else 'Low'
            }

            # 创建文本显示
            stats_text = "\n".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                                    for k, v in stats.items()])

            ax4.text(0.1, 0.5, stats_text, fontsize=12,
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax4.axis('off')
            ax4.set_title('Current Emotion Statistics')

        plt.suptitle('Real-time Emotion Analysis Dashboard', fontsize=16, y=0.98)
        plt.tight_layout()

        # 转换为PIL图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img