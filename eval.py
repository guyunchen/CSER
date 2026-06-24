import time
import torch
from thop import profile
from modules.ls4_CTLN import LiteGLSER

# 假设你的模型叫 LiteGLSER
# from model import LiteGLSER
# model = LiteGLSER()

# 示例：先创建模型实例
model = LiteGLSER()
model.eval()

# ===============================
# 1️⃣ FLOPs & Params
# ===============================

# 假设输入为 batch=1, 300帧, 80维log-mel
dummy_input = torch.randn(1, 300, 80)

flops, params = profile(
    model,
    inputs=(dummy_input,)
)

print(f"FLOPs: {flops / 1e6:.2f} MFLOPs")
print(f"Params: {params / 1e6:.2f} M")

# ===============================
# 2️⃣ Latency 测量 (ms)
# ===============================

# warmup
for _ in range(20):
    _ = model(dummy_input)

torch.cuda.synchronize() if torch.cuda.is_available() else None

# 测量多次平均
n_runs = 100
start_time = time.time()
for _ in range(n_runs):
    _ = model(dummy_input)
torch.cuda.synchronize() if torch.cuda.is_available() else None
end_time = time.time()

latency_ms = (end_time - start_time) / n_runs * 1000
print(f"Latency: {latency_ms:.2f} ms per sample")

# ===============================
# 3️⃣ RTF (Real-Time Factor)
# ===============================

# 假设输入音频实际时长为 audio_duration 秒
audio_duration = 3.0  # 3秒音频
rtf = (latency_ms / 1000.0) / audio_duration
print(f"RTF: {rtf:.4f} (smaller than 1 means faster than real-time)")
