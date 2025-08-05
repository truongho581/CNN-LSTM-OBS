import os
import re
import random
import glob
import numpy as np
from obspy import read, UTCDateTime
from seisbench.models.pickblue import PickBlue
from scipy.signal import spectrogram
from PIL import Image
from pathlib import Path

# ==== Config ====
BASE_DIR = "data"
MSEED_DIR = "mseed_data"
TARGET_SIZE = (64, 64)
NUM_FRAMES = 30
FRAME_LEN = 2.0  # seconds
model = PickBlue(base="eqtransformer")

# ==== Spectrogram ====
def make_spectrogram(data, fs, fmin=2, fmax=10):
    nperseg = int(0.2* fs)
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_mask = (f >= fmin) & (f <= fmax)
    Sxx = Sxx[f_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    Sxx_db = np.clip(Sxx_db, -80, 60)
    Sxx_norm = (Sxx_db + 80) / 140
    img = Image.fromarray((Sxx_norm * 255).astype(np.uint8))
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

# ==== Chia frame ====
def split_into_frames(data, fs):
    spf = int(FRAME_LEN * fs)
    return [data[i * spf:(i + 1) * spf] for i in range(NUM_FRAMES)]

# ==== Process EQ sample ====
def process_eq_sample(station, index):
    path = f"{MSEED_DIR}/{station}/{station}_eq_{index:03}.mseed"
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    st = read(path)
    fs = st[0].stats.sampling_rate
    trZ = st.select(channel="BHZ")[0]
    trN = st.select(channel="BHN")[0]
    trE = st.select(channel="BHE")[0]

    result = model.classify(st)
    p_pick_time = None
    for pick in result.picks:
        if str(pick).strip().endswith("P"):
            match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", str(pick))
            if match:
                p_pick_time = UTCDateTime(match.group(0))
                break

    if not p_pick_time:
        print(f"⚠️ Không tìm thấy PickBlue P-onset cho {station}_eq_{index:03}")
        return

    # ✅ Tính offset chính xác theo giây và frame
    offset_sec = random.uniform(0, 10)
    start_cut = p_pick_time - offset_sec
    end_cut = start_cut + NUM_FRAMES * FRAME_LEN

    trZ = trZ.trim(start_cut, end_cut, pad=True, fill_value=0)
    trN = trN.trim(start_cut, end_cut, pad=True, fill_value=0)
    trE = trE.trim(start_cut, end_cut, pad=True, fill_value=0)

    Z_frames = split_into_frames(trZ.data, fs)
    N_frames = split_into_frames(trN.data, fs)
    E_frames = split_into_frames(trE.data, fs)

    spec_frames = []
    for z, n, e in zip(Z_frames, N_frames, E_frames):
        specZ = make_spectrogram(z, fs)
        specN = make_spectrogram(n, fs)
        specE = make_spectrogram(e, fs)
        spec_rgb = np.stack([specZ, specN, specE], axis=-1)
        spec_frames.append(spec_rgb)

    spec_seq = np.stack(spec_frames, axis=0)

    # ✅ Gán nhãn chuẩn từ frame chứa P-onset
    offset_frame = int(offset_sec // FRAME_LEN)
    offset_frame = min(offset_frame, NUM_FRAMES)
    labels = [0] * offset_frame + [1] * (NUM_FRAMES - offset_frame)

    out_dir = f"{BASE_DIR}/{station}/eq_{index:03}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "sample.npz"), spec=spec_seq, fs=fs, p_arrival=p_pick_time.timestamp)
    with open(os.path.join(out_dir, "label.txt"), "w") as f:
        f.write(" ".join(map(str, labels)))

    print(f"✅ EQ saved: {station}_eq_{index:03} (offset_frame={offset_frame})")


# ==== Process noise sample từ file path ====
def process_noise_file(filepath, station, index):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    st = read(filepath)
    fs = st[0].stats.sampling_rate
    trZ = st.select(channel="BHZ")[0]
    trN = st.select(channel="BHN")[0]
    trE = st.select(channel="BHE")[0]

    total_samples = len(trZ.data)
    samples_30s = int(NUM_FRAMES * FRAME_LEN * fs)

    if total_samples < samples_30s:
        print(f"⚠️ Noise quá ngắn: {Path(filepath).name}")
        return

    max_start = total_samples - samples_30s
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + samples_30s

    Z_frames = split_into_frames(trZ.data[start_idx:end_idx], fs)
    N_frames = split_into_frames(trN.data[start_idx:end_idx], fs)
    E_frames = split_into_frames(trE.data[start_idx:end_idx], fs)

    spec_frames = []
    for z, n, e in zip(Z_frames, N_frames, E_frames):
        specZ = make_spectrogram(z, fs)
        specN = make_spectrogram(n, fs)
        specE = make_spectrogram(e, fs)
        spec_rgb = np.stack([specZ, specN, specE], axis=-1)
        spec_frames.append(spec_rgb)

    spec_seq = np.stack(spec_frames, axis=0)
    out_dir = f"{BASE_DIR}/{station}/noise_{index:03}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "sample.npz"), spec=spec_seq, fs=fs)
    with open(os.path.join(out_dir, "label.txt"), "w") as f:
        f.write(" ".join(["0"] * NUM_FRAMES))

    print(f"🎵 Noise saved: {Path(filepath).name} → noise_{index:03}")

# ==== Duyệt toàn bộ noise file ====
def process_all_noise_for_station(station):
    noise_files = sorted(glob.glob(f"{MSEED_DIR}/{station}/{station}_noise_*.mseed"))
    for i, path in enumerate(noise_files):
        print(f"🔁 Processing noise: {Path(path).name}")
        process_noise_file(path, station, i)

# ==== Duyệt toàn bộ EQ ====
def process_all_eq_for_station(station):
    eq_files = sorted(glob.glob(f"{MSEED_DIR}/{station}/{station}_eq_*.mseed"))
    for path in eq_files:
        match = re.search(rf"{station}_eq_(\d+)\.mseed", Path(path).name)
        if match:
            index = int(match.group(1))
            process_eq_sample(station, index)

# ==== Duyệt tất cả trạm trong mseed_data/ ====
def process_all_stations():
    stations = [d.name for d in Path(MSEED_DIR).iterdir() if d.is_dir()]
    print(f"🔍 Found {len(stations)} station(s): {stations}")
    for station in stations:
        print(f"\n📡 Processing station: {station}")
        process_all_eq_for_station(station)
        process_all_noise_for_station(station)

# ==== Entry point ====
if __name__ == "__main__":
    process_all_stations()
