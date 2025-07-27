import obspy
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
from tensorflow.keras.models import load_model
from PIL import Image

# ==== CONFIG ====
MODEL_PATH = "best_model.h5"
MSEED_FILE = "waveforms/M12B_event_000.mseed"
IMG_SIZE = (64, 64)
FRAME_LENGTH = 5.0
BANDPASS = (1.0, 10.0)
THRESHOLD = 0.5
MAX_FRAMES = 30   # lấy 30 frame đầu tiên

# ==== 1. Load waveform ====
st = obspy.read(MSEED_FILE)
tr = st[0]
tr.detrend("demean")
tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1])

fs = tr.stats.sampling_rate
segment_samples = int(FRAME_LENGTH * fs)

# ==== 2. Hàm segment -> spectrogram ====
def segment_to_spectrogram(segment, fs):
    nperseg = int(5.12 * fs)
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)

    # chỉ giữ dải tần 2–10 Hz
    f_mask = (f >= 2) & (f <= 10)
    power_db = power_db[f_mask, :]

    # chuẩn hóa 0–1
    power_db = np.clip((power_db - power_db.min()) /
                       (power_db.max() - power_db.min() + 1e-8), 0, 1)

    # chuyển về ảnh xám và resize
    img = Image.fromarray((power_db * 255).astype(np.uint8)).convert("L")
    img = img.resize(IMG_SIZE)

    # về numpy float32 và chuẩn hóa 0–1
    img = np.array(img).astype(np.float32) / 255.0

    # thêm channel dimension (H,W,1)
    return img[..., np.newaxis]

# ==== 3. Cắt 30 frame đầu tiên ====
frames = []
for i in range(MAX_FRAMES):  # chỉ lấy 30 frame
    start = i * segment_samples
    end = start + segment_samples
    if end > len(tr.data):
        break
    seg = tr.data[start:end]
    img = segment_to_spectrogram(seg, fs)
    frames.append(img)

frames = np.array(frames)
print(f"✅ Đã lấy {frames.shape[0]} frame đầu tiên")

# ==== 4. Pad nếu thiếu để thành 30 ====
if frames.shape[0] < MAX_FRAMES:
    pad_len = MAX_FRAMES - frames.shape[0]
    pad = np.zeros((pad_len, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)
    frames = np.concatenate([frames, pad], axis=0)

# ==== 5. Dự đoán ====
sequence = np.expand_dims(frames, axis=0)  # (1,30,64,64,1)
model = load_model(MODEL_PATH)
y_pred = model.predict(sequence, verbose=0)[0][:,0]

# ==== 6. Xuất nhãn ====
labels = (y_pred > THRESHOLD).astype(int)
label_string = "".join(str(x) for x in labels)
print("🔍 Nhãn 30 frame đầu:", label_string)
print("🔍 Xác suất:", y_pred)
