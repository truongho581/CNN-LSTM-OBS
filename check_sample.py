import numpy as np
import matplotlib.pyplot as plt

# === Cấu hình ===
npz_path = "data/FS01B/eq_002/sample.npz"
NUM_FRAMES = 30
SAVE_PATH = "figure_rgb_frames_compact.png"

# === Load dữ liệu ===
data = np.load(npz_path)
spec_seq = data['spec']

# === Vẽ 20 frames RGB với spacing nhỏ ===
n_cols = 5
n_rows = (NUM_FRAMES + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

# 👉 Giảm khoảng cách giữa các subplot
plt.subplots_adjust(wspace=0.05, hspace=0.2)

for i in range(n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    if i < NUM_FRAMES:
        ax.imshow(spec_seq[i])
    ax.axis("off")

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"✅ Saved compact figure to {SAVE_PATH}")
plt.close()
