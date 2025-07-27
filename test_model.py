import obspy
import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
from tensorflow.keras.models import load_model
from PIL import Image

# ==== CONFIG ====
MODEL_PATH = "drive/MyDrive/OBS/best_model.h5"
MSEED_FILE = "long_data.mseed"
IMG_SIZE = (64, 64)
FRAME_LENGTH = 5.0   # giây mỗi frame
BANDPASS = (1.0, 10.0)
THRESHOLD = 0.5      # ngưỡng phân loại EQ/Noise

# ==== 1. Load waveform ====
st = obspy.read(MSEED_FILE)
tr = st[0]
tr.detrend("demean")
tr.filter("bandpass", freqmin=BANDPASS[0], freqmax=BANDPASS[1])

fs = tr.stats.sampling_rate
segment_samples = int(FRAME_LENGTH * fs)

# ==== 2. Segment -> spectrogram ====
def segment_to_spectrogram(segment, fs, img_size=(64,64)):
    f, t, Sxx = spectrogram(segment, fs=fs, nperseg=256, noverlap=128)
    power_db = 10 * np.log10(Sxx + 1e-10)
    f_mask = (f >= 2) & (f <= 10)
    power_db = power_db[f_mask, :]
    # chuẩn hóa 0-1
    power_db = np.clip((power_db - power_db.min()) / (power_db.max() - power_db.min() + 1e-8), 0, 1)
    img = Image.fromarray((power_db*255).astype(np.uint8))
    img = img.resize(img_size)
    img = np.array(img).astype(np.float32) / 255.0
    return img[..., np.newaxis]

# ==== 3. Cắt thành các frame 5s ====
frames = []
for i in range(0, len(tr.data)-segment_samples, segment_samples):
    seg = tr.data[i:i+segment_samples]
    img = segment_to_spectrogram(seg, fs, IMG_SIZE)
    frames.append(img)

frames = np.array(frames)  # (num_frames, H, W, 1)
num_frames = frames.shape[0]
print(f"✅ Tổng số frame: {num_frames}")

# ==== 4. Dự đoán với model ====
model = load_model(MODEL_PATH)

# Cho toàn bộ thành một sequence dài -> mô hình trả về xác suất từng frame
sequence = np.expand_dims(frames, axis=0)  # (1, num_frames, 64, 64, 1)
y_pred = model.predict(sequence, verbose=1)[0][:,0]  # vector xác suất (num_frames,)

# ==== 5. Chuyển thành chuỗi 0/1 ====
labels = (y_pred > THRESHOLD).astype(int)
label_string = "".join(str(x) for x in labels)

print(f"🔍 Chuỗi nhãn dự đoán:\n{label_string}")
