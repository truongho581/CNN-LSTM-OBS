import os
import random
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.signal import spectrogram
from PIL import Image
import matplotlib.pyplot as plt
from event_info import radius, minmagni
import shutil


client = Client("IRIS")

# ==== Danh sách trạm với thời gian riêng ====
stations_list = [
    # ['FS01B', 40.3268, -124.9492, -940.0, '2012-09-02', '2013-06-19'],
    # ['FS02D', 40.3260, -124.8002, -947.9, '2014-07-16', '2015-08-28'],
    # ['FS10D', 40.4328, -124.6940, -1153.8, '2014-07-17', '2015-08-27'],
    # ['FS16D', 40.5378, -124.7468, -1080.4, '2014-07-16', '2015-08-28'],
    # ['FS41D', 40.6124, -124.7310, -1079.3, '2014-07-16', '2015-08-28'],
    # ['FS44D', 40.7609, -124.7028, -837.0, '2014-08-12', '2015-09-15'],
    # ['G01D', 39.9999, -124.6008, -1006.7, '2014-07-17', '2015-08-27'],
    # ['M11B', 42.9320, -125.0171, -1109.0, '2012-09-02', '2013-06-18'],
    # ['G09D', 40.6665, -124.7269, -716.3, '2014-07-16', '2015-08-28'],
    # ['FS04D', 40.2528, -124.5052, -155.0, '2014-08-12', '2015-09-15'],
    # ['FS08D', 40.3347, -124.4653, -132.0, '2014-08-12', '2015-09-15'],
    # ['FS14B', 40.4955, -124.5918, -107.0, '2012-09-02', '2013-06-19'],
    # ['J09B', 43.1510, -124.7270, -252.0, '2012-09-02', '2013-06-21'],
    # ['J25A', 44.4729, -124.6216, -142.8, '2011-10-21', '2012-07-18'],
    # ['J25C', 44.4730, -124.6217, -144.0, '2013-08-22', '2014-06-01'],
    # ['J33C', 45.1068, -124.5708, -354.0, '2013-08-22', '2014-05-29'],
    # ['J65C', 47.8913, -125.1398, -169.0, '2013-08-19', '2014-05-29'],
    # ['J73A', 48.7677, -126.1925, -143.3, '2011-10-18', '2012-07-17'],
    # ['J73C', 48.7679, -126.1926, -133.0, '2013-08-19', '2014-05-30'],
    # ['M01A', 49.1504, -126.7221, -132.9, '2011-10-18', '2012-07-16'],
    # ['M01C', 49.1504, -126.7222, -138.0, '2013-08-19', '2014-05-30'],
    # ['M02A', 48.3070, -125.6004, -139.0, '2011-10-18', '2012-07-16'],

    ['M12B', 42.1840, -124.9461, -1045.0, '2012-09-02', '2013-06-18'],
    ['M14B', 40.9850, -124.5897, -638.0, '2012-09-02', '2013-06-19'],
    ['M16D', 41.6618, -124.8071, -882.0, '2014-08-12', '2015-09-17'],
    ['G33D', 42.6653, -124.8020, -686.0, '2014-08-12', '2015-09-17'],
    ['M02C', 48.3069, -125.6012, -141.0, '2013-08-19', '2014-05-30'],
    ['M08A', 44.1187, -124.8953, -126.4, '2011-10-20', '2012-07-18'],
    ['M08C', 44.1185, -124.8954, -131.0, '2013-08-21', '2014-06-01'],
    ['M04A', 47.5581, -125.1922, -563.4, '2011-10-17', '2012-07-16'],
    ['M04C', 47.5584, -125.1923, -570.0, '2013-08-19', '2014-05-29'],
    ['M05A', 46.1735, -124.9346, -828.2, '2011-10-17', '2012-07-15'],
    ['M05C', 46.1735, -124.9345, -837.0, '2013-08-19', '2014-05-29'],
]


NETWORK = "7D"
CHANNEL = "BHZ"
LOCATION = "--"

# ==== Config ====
os.makedirs("samples_test", exist_ok=True)
os.makedirs("waveforms", exist_ok=True)
os.makedirs("full_spectrogram", exist_ok=True)

VMIN = -80
VMAX = 60
MAX_FRAMES = 40

# ==== Hàm vẽ full spectrogram ====
def plot_full_spectrogram(tr, station, index):
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
    plt.title(f"Full Spectrogram - {station} Event {index}")
    plt.tight_layout()
    plt.savefig(f"full_spectrogram/{station}_eq_{index:03}_full.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

# ==== Hàm tạo frame 64×64 gray scale ====
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

# ==== Vòng lặp qua trạm ====
for station_data in stations_list:
    STATION, lat, lon, elev, t_start, t_end = station_data
    start_time = UTCDateTime(t_start)
    end_time = UTCDateTime(t_end)

    print(f"\n=== 📡 Đang xử lý trạm {STATION} ({t_start} → {t_end}) ===")

    catalog = client.get_events(starttime=start_time,
                                endtime=end_time,
                                latitude=lat,
                                longitude=lon,
                                maxradius=radius,
                                minmagnitude=minmagni)
    catalog.events.sort(key=lambda e: e.origins[0].time)
    print(f"---🔍 {STATION}: {len(catalog)} sự kiện---")

    for i, event in enumerate(catalog):
        seq_dir = f"samples_test/{STATION}_sample_{i:03}"
        try:
            origin_time = event.origins[0].time
            print(f"\n🔍 Event {i}: {origin_time}")

            st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL,
                                      starttime=origin_time - 60,
                                      endtime=origin_time + 420)
            st.detrend("demean")
            st.filter("bandpass", freqmin=1, freqmax=10)
            tr = st[0]

            plot_full_spectrogram(tr, STATION, i)

            fs = tr.stats.sampling_rate
            segment_length = 5
            segment_samples = int(segment_length * fs)

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
            trigger_end = None

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

                            # 🔹 Random pre-noise 1–5 frame
                            n_pre = random.randint(1, 5)
                            pre_start = max(0, trigger_start - n_pre)
                            for k in range(pre_start, trigger_start):
                                seg = tr.data[k*segment_samples:(k+1)*segment_samples]
                                if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                                    labels.append(0)

                            # 3 frame trigger = EQ
                            for k in range(trigger_start, j+1):
                                seg = tr.data[k*segment_samples:(k+1)*segment_samples]
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
                    seg = tr.data[j*segment_samples:(j+1)*segment_samples]
                    if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                        labels.append(1)
                    trigger_end = j
                    if labels.count(1) >= max_frame_after_trigger:
                        break

            n_eq = labels.count(1)
            n_noise = labels.count(0)

            # Nếu không có EQ thì xóa folder
            if n_eq == 0:
                print(f"⚠️ {STATION} Event {i} không có EQ, xóa folder")
                shutil.rmtree(seq_dir, ignore_errors=True)
                continue

            # Bù noise để cân bằng
            diff = n_eq - n_noise
            if diff > 0 and trigger_end is not None:
                post_start = trigger_end + 1
                for k in range(post_start, post_start+diff):
                    if k >= len(avg_power_list):
                        break
                    seg = tr.data[k*segment_samples:(k+1)*segment_samples]
                    if segment_to_image(seg, fs, f"{seq_dir}/frame_{len(labels):03}.png"):
                        labels.append(0)

            # Giới hạn MAX_FRAMES (không pad)
            if len(labels) > MAX_FRAMES:
                labels = labels[:MAX_FRAMES]
                for fname in os.listdir(seq_dir):
                    if fname.startswith("frame_"):
                        idx = int(fname.split("_")[1].split(".")[0])
                        if idx >= MAX_FRAMES:
                            os.remove(os.path.join(seq_dir, fname))

            with open(f"{seq_dir}/label.txt", "w") as f:
                f.write("".join(str(x) for x in labels))

            print(f"🎉 {STATION} Event {i} done. EQ={n_eq}, Noise={labels.count(0)}, Total={len(labels)}")

        except Exception as e:
            print(f"❌ Error {STATION} Event {i}: {e}")
            shutil.rmtree(seq_dir, ignore_errors=True)