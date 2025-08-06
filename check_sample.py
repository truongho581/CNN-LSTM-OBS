import numpy as np
import matplotlib.pyplot as plt

# === Cấu hình ===
npz_path = "data/FS01B/eq_005/sample.npz"  # 🔁 Thay đường dẫn tại đây nếu cần
NUM_FRAMES = 30

# === Tải dữ liệu ===
data = np.load(npz_path)
spec_seq = data['spec']  # shape: (30, 64, 64, 3)

# === Kiểm tra shape ===
print(f"Shape: {spec_seq.shape} (frames, height, width, channels)")

# === Vẽ 30 frames RGB ===
n_cols = 6
n_rows = (NUM_FRAMES + n_cols - 1) // n_cols
plt.figure(figsize=(15, 8))
for i in range(NUM_FRAMES):
    plt.subplot(n_rows, n_cols, i+1)
    plt.imshow(spec_seq[i])
    plt.axis("off")
    plt.title(f"Frame {i}")
plt.tight_layout()
plt.show()
