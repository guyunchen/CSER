import random
import torch
import torch.nn as nn


class FeatureAugmentor(nn.Module):

    def __init__(
        self,
        aug_prob=0.3,
        time_mask_param=8,
        freq_mask_param=4,
        noise_std=0.003,
        max_shift=5
    ):
        super().__init__()

        self.aug_prob = aug_prob

        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        self.noise_std = noise_std
        self.max_shift = max_shift

    def time_mask(self, x):
        """
        x: [T, F]
        """

        if random.random() > 0.5:
            return x

        T = x.size(0)

        if T <= 1:
            return x

        mask_len = random.randint(1, self.time_mask_param)
        mask_len = min(mask_len, T - 1)

        start = random.randint(0, T - mask_len)

        x[start:start + mask_len, :] = 0

        return x

    def freq_mask(self, x):
        """
        x: [T, F]
        """

        if random.random() > 0.5:
            return x

        Freq = x.size(1)

        mask_len = random.randint(1, self.freq_mask_param)
        mask_len = min(mask_len, Freq - 1)

        start = random.randint(0, Freq - mask_len)

        x[:, start:start + mask_len] = 0

        return x

    def temporal_shift(self, x):
        """
        x: [T, F]
        """

        if random.random() > 0.5:
            return x

        shift = random.randint(
            -self.max_shift,
            self.max_shift
        )

        x = torch.roll(x, shifts=shift, dims=0)

        return x

    def add_noise(self, x):
        """
        Very lightweight Gaussian noise
        """

        if random.random() > 0.3:
            return x

        noise = torch.randn_like(x) * self.noise_std

        return x + noise

    def forward(self, x):

        if random.random() > self.aug_prob:
            return x

        x = x.clone()

        x = self.time_mask(x)
        x = self.freq_mask(x)
        x = self.temporal_shift(x)
        x = self.add_noise(x)

        return x