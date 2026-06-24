import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelRefiner(nn.Module):
    """
    针对回归任务的标签处理模块：
    1. Label Delay (标签移位): 补偿人类标注的反应时间延迟
    2. Label Smoothing (高斯平滑): 减少标注噪声
    """
    def __init__(self, delay_steps=0, sigma=0.5, apply_smoothing=True):
        super().__init__()
        self.delay_steps = delay_steps
        self.sigma = sigma
        self.apply_smoothing = apply_smoothing

    def forward(self, labels):
        """
        labels: [Batch, 2] (Valence, Arousal)
        注：如果你的数据是 utterance-level (每个段落一个值)，
        'Delay' 通常体现为引入极小量的随机扰动或基于置信度的平滑。
        """
        refined_labels = labels.clone()

        # 1. 高斯平滑/标签噪声处理 (针对回归的 Label Smoothing)
        if self.apply_smoothing and self.training:
            # 在回归目标上加上微小的高斯噪声，防止模型过拟合于某一个具体的标注点
            noise = torch.randn_like(refined_labels) * (self.sigma * 0.1)
            refined_labels = refined_labels + noise
            # 确保不越界 (假设标签已经 sigmoid 或归一化到 0-1)
            refined_labels = torch.clamp(refined_labels, 0.0, 1.0)

        return refined_labels