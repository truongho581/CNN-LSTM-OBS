import os
import random
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.signal import spectrogram
from OBS_station import start_time, end_time, client, LOCATION, NETWORK, CHANNEL, station_data
from event_info import radius, minmagni
from PIL import Image
import matplotlib.pyplot as plt

# ==== Config ====
os.makedirs("waveforms", exist_ok=True)
os.makedirs("samples", exist_ok=True)
os.makedirs("full_spectrogram", exist_ok=True)

STATION, lat, lon, elev = station_data
VMIN = -80
VMAX = 60
MAX_FRAMES = 40

# ==== Lấy catalog ====
catalog = client.get_events(starttime=start_time,
                            endtime=end_time,
                            latitude=lat,
                            longitude=lon,
                            maxradius=radius,
                            minmagnitude=minmagni)
catalog.events.sort(key=lambda e: e.origins[0].time)
print(f"---🔍 Tổng cộng {len(catalog)} sự kiện---")

# ==== Hàm vẽ full spectrogram ====
def plot_full_spectrogram(tr, label, index):
    fs = tr.stats.sampling_rate
    nperseg = int(5.12 * fs)
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(tr.data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)
    f_mask = (f >= 2) & (f <= 10)
    f = f[f_mask]
    power_db = power_db[f_mask, :]
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, power_db, shading='gouraud', cmap='inferno', vmin=VMIN, vmax=VMAX)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.title(f"Full Spectrogram - {label} {index}")
    plt.tight_layout()
    plt.savefig(f"full_spectrogram/{STATION}_{label.lower()}_{index:03}_full.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

# ==== Hàm tạo frame gray scale 64×64 với scale cố định ====
def segment_to_image(segment, fs, save_path):
    safe_nperseg = min(256, int(len(segment) * 0.5))
    safe_noverlap = int(0.5 * safe_nperseg)
    f, t, Sxx = spectrogram(segment, fs=fs, nperseg=safe_nperseg, noverlap=safe_noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)
    f_mask = (f >= 2) & (f <= 10)
    power_db = power_db[f_mask, :]

    if np.max(power_db) < -40:
        return False

    power_db = np.clip((power_db - VMIN) / (VMAX - VMIN), 0, 1)
    img = Image.fromarray((power_db * 255).astype(np.uint8)).convert("L")
    img = img.resize((64, 64))
    img.save(save_path)
    return True

# ==== Xử lý từng event ====
for i, event in enumerate(catalog):
    try:
        origin_time = event.origins[0].time
        print(f"\n🔍 Event {i}: {origin_time}")

        st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL,
                                  starttime=origin_time - 60,
                                  endtime=origin_time + 420)
        st.detrend("demean")
        st.filter("bandpass", freqmin=1, freqmax=10)
        tr = st[0]

        plot_full_spectrogram(tr, "EQ", i)

        fs = tr.stats.sampling_rate
        segment_length = 5
        segment_samples = int(segment_length * fs)

        # Năng lượng frame
        avg_power_list = []
        for j in range(200):
            start_idx = j * segment_samples
            end_idx = start_idx + segment_samples
            if end_idx > len(tr.data):
                break
            seg = tr.data[start_idx:end_idx]
            power_db = 10 * np.log10(np.mean(seg**2) + 1e-10)
            avg_power_list.append(power_db)

        pre_event_idx = int(60 / segment_length)
        noise_mean = np.mean(avg_power_list[:pre_event_idx])
        trigger_threshold = noise_mean + 6
        margin = 6
        trigger_window = 3
        max_frame_after_trigger = 40

        labels = []
        triggered = False
        window_count = 0
        weak_count = 0
        trigger_start = None
        trigger_end = None

        seq_dir = f"samples/{STATION}_sample_{i:03}"
        os.makedirs(seq_dir, exist_ok=True)

        for j in range(12, len(avg_power_list)):
            avg_power = avg_power_list[j]

            if not triggered:
                if avg_power > trigger_threshold:
                    window_count += 1
                    if window_count >= trigger_window:
                        triggered = True
                        trigger_start = j - (trigger_window - 1)
                        print(f"✅ Trigger ON tại seg {j} (start={trigger_start})")

                        # 🔹 Random số frame pre-noise 2–5
                        n_pre = random.randint(2, 5)
                        pre_start = max(0, trigger_start - n_pre)
                        for k in range(pre_start, trigger_start):
                            start_idx = k * segment_samples
                            end_idx = start_idx + segment_samples
                            seg = tr.data[start_idx:end_idx]
                            if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                                labels.append(0)

                        # 3 frame trigger = EQ
                        for k in range(trigger_start, j+1):
                            start_idx = k * segment_samples
                            end_idx = start_idx + segment_samples
                            seg = tr.data[start_idx:end_idx]
                            if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                                labels.append(1)
                        trigger_end = j
                else:
                    window_count = 0
                continue

            if avg_power <= (noise_mean + margin):
                weak_count += 1
                if weak_count >= 3:
                    print(f"⏹️ Trigger OFF tại seg {j}")
                    break
            else:
                weak_count = 0
                start_idx = j * segment_samples
                end_idx = start_idx + segment_samples
                seg = tr.data[start_idx:end_idx]
                if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                    labels.append(1)
                trigger_end = j
                if labels.count(1) >= max_frame_after_trigger:
                    break

        n_eq = labels.count(1)
        n_noise = labels.count(0)
        diff = n_eq - n_noise
        if diff > 0 and trigger_end is not None:
            post_start = trigger_end + 1
            for k in range(post_start, post_start+diff):
                if k >= len(avg_power_list):
                    break
                start_idx = k * segment_samples
                end_idx = start_idx + segment_samples
                seg = tr.data[start_idx:end_idx]
                if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                    labels.append(0)

        # === Giới hạn về MAX_FRAMES ===
        if len(labels) > MAX_FRAMES:
            labels = labels[:MAX_FRAMES]
            for fname in os.listdir(seq_dir):
                if fname.startswith("frame_"):
                    idx = int(fname.split("_")[1].split(".")[0])
                    if idx >= MAX_FRAMES:
                        os.remove(os.path.join(seq_dir, fname))
        elif len(labels) < MAX_FRAMES:
            pad_len = MAX_FRAMES - len(labels)
            for p in range(pad_len):
                img = Image.new("L", (64, 64), color=0)
                img.save(f"{seq_dir}/frame_{len(labels)+p:03}.png")
                labels.append(0)

        with open(f"{seq_dir}/label.txt", "w") as f:
            f.write("".join(str(x) for x in labels))

        print(f"🎉 Event {i} done. EQ={n_eq}, Noise={labels.count(0)}")

    except Exception as e:
        print(f"❌ Error Event {i}: {e}")
