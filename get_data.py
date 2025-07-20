import os
import random
import numpy as np
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.signal import spectrogram
from OBS_station import start_time, end_time, client, LOCATION, NETWORK, CHANNEL, station_data
from event_info import radius, minmagni

os.makedirs("waveforms", exist_ok=True)
os.makedirs("earthquake", exist_ok=True)
os.makedirs("noise", exist_ok=True)
os.makedirs("full_spectrogram", exist_ok=True)

STATION, lat, lon, elev = station_data

catalog = client.get_events(starttime=start_time,
                            endtime=end_time,
                            latitude=lat,
                            longitude=lon,
                            maxradius=radius,
                            minmagnitude=minmagni)

# Sắp xếp theo thời gian tăng dần
catalog.events.sort(key=lambda e: e.origins[0].time)

print(f"---🔍 Tổng cộng {len(catalog)} sự kiện---")

# Tạo vùng cấm ±10 phút
no_go_intervals = [(e.origins[0].time - 600, e.origins[0].time + 600) for e in catalog]

def is_interval_safe(start, duration=180):
    return all(start + duration < ban_start or start > ban_end for ban_start, ban_end in no_go_intervals)


# === Hàm vẽ spectrogram FULL waveform ===
def plot_full_spectrogram(tr, label, index):
    fs = tr.stats.sampling_rate
    nperseg = int(5.12 * fs)  # ví dụ: 5.12s → 256 tại fs=50Hz
    noverlap = int(0.5 * nperseg)

    f, t, Sxx = spectrogram(tr.data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)
    f_mask = (f >= 2) & (f <= 10)
    f = f[f_mask]
    power_db = power_db[f_mask, :]

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, power_db, shading='gouraud',
                   cmap='inferno', vmin=-80, vmax=60)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")
    plt.title(f"Full Spectrogram - {label} {index}")
    plt.tight_layout()
    plt.savefig(f"full_spectrogram/{STATION}_{label.lower()}_{index:03}_full.png",
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


# === Hàm vẽ 1 frame ===
def plot_frame_spectrogram_fixed(segment, fs, segment_length, save_path):
    safe_nperseg = min(256, int(len(segment) * 0.5))
    safe_noverlap = int(0.5 * safe_nperseg)

    f, t, Sxx = spectrogram(segment, fs=fs, nperseg=safe_nperseg, noverlap=safe_noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)

    f_mask = (f >= 2) & (f <= 10)  # 👈 chỉ giữ tần số 2–10Hz
    f = f[f_mask]
    power_db = power_db[f_mask, :]

    if np.max(power_db) < -40:
        return False  # 👈 bỏ nếu quá yên tĩnh

    fig, ax = plt.subplots(figsize=(6, 4))
    t_uniform = np.linspace(0, segment_length, power_db.shape[1])
    ax.pcolormesh(t_uniform, f, power_db, shading='gouraud',
                  cmap='inferno', vmin=-80, vmax=60)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return True

# === LOOP XỬ LÝ 1 EVENT ===
for i, event in enumerate(catalog):
    try:
        origin_time = event.origins[0].time
        print(f"\n🔍 Đang xử lý Event {i}: {origin_time}")

        st = client.get_waveforms(
            network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL,
            starttime=origin_time - 60,
            endtime=origin_time + 420
        )
        st.detrend("demean")
        st.filter("bandpass", freqmin=1, freqmax=10)
        st.write(f"waveforms/{STATION}_event_{i:03}.mseed", format="MSEED")

        tr = st[0]
        plot_full_spectrogram(tr, "EQ", i)

        fs = tr.stats.sampling_rate
        segment_length = 5  # giây
        segment_samples = int(segment_length * fs)

        # === Tính năng lượng trung bình toàn waveform ===
        avg_power_list = []
        for j in range(200):
            start_idx = j * segment_samples
            end_idx = start_idx + segment_samples
            if end_idx > len(tr.data):
                break

            segment = tr.data[start_idx:end_idx]
            energy = np.mean(segment ** 2)
            power_db = 10 * np.log10(energy + 1e-10)
            avg_power_list.append(power_db)

            print(f"[EQ {i}, Segment {j}] Avg Power: {power_db:.2f} dB")

        # === Tính noise_mean ===
        pre_event_idx = int(60 / segment_length)
        noise_only = avg_power_list[:pre_event_idx]
        noise_mean = np.mean(noise_only)
        print(f"📊 Noise mean (trước origin_time): {noise_mean:.2f} dB")

        # === Tham số trigger ===
        trigger_threshold = noise_mean + 6
        margin = 6
        trigger_window = 3
        max_frame_after_trigger = 40  # 🔴 Giới hạn frame sau trigger

        triggered = False
        window_count = 0
        weak_count = 0
        frame_count = 0

        seq_dir = f"earthquake/{STATION}_sample_{i:03}"

        for j in range(12, len(avg_power_list)):
            avg_power = avg_power_list[j]

            # === Trigger ON ===
            if not triggered:
                if avg_power > trigger_threshold:
                    window_count += 1
                    if window_count >= trigger_window:
                        triggered = True
                        trigger_start = j - (trigger_window - 1)
                        print(f"✅ Trigger ON tại segment {j} ➜ lưu từ segment {trigger_start}")

                        for k in range(trigger_start, j + 1):
                            if k < 0: continue
                            start_idx = k * segment_samples
                            end_idx = start_idx + segment_samples
                            segment = tr.data[start_idx:end_idx]

                            if frame_count == 0:
                                os.makedirs(seq_dir, exist_ok=True)

                            frame_count += 1
                            if frame_count > max_frame_after_trigger:
                                print(f"🛑 Đã đủ {max_frame_after_trigger} frame, dừng lưu.")
                                break

                            frame_name = f"{seq_dir}/frame_{frame_count:03}.png"
                            plot_frame_spectrogram_fixed(segment, fs, segment_length, frame_name)
                else:
                    window_count = 0
                continue

            # === Nếu đã Trigger ON ===
            if avg_power < (noise_mean + margin):
                weak_count += 1
                print(f"⚠️ Segment {j} yếu ({avg_power:.2f} dB) → weak_count={weak_count}")
                if weak_count >= 3:
                    print(f"⏹️ Dừng tại segment {j} vì gặp {weak_count} seg yếu liên tiếp")
                    break
            else:
                weak_count = 0
                start_idx = j * segment_samples
                end_idx = start_idx + segment_samples
                if end_idx > len(tr.data):
                    break

                segment = tr.data[start_idx:end_idx]

                if frame_count == 0:
                    os.makedirs(seq_dir, exist_ok=True)

                frame_count += 1
                if frame_count > max_frame_after_trigger:
                    print(f"🛑 Đã đủ {max_frame_after_trigger} frame, dừng lưu.")
                    break

                frame_name = f"{seq_dir}/frame_{frame_count:03}.png"
                plot_frame_spectrogram_fixed(segment, fs, segment_length, frame_name)

        if frame_count == 0:
            print(f"⚠️ Event {i} không đủ tín hiệu mạnh.")
            continue

        print(f"🎉 Event {i} xong. Đã lưu {frame_count} frame ➜ {seq_dir}")

    except Exception as e:
        print(f"❌ Lỗi với Event {i}: {e}")



# === LẤY NOISE ===
num_eq_samples = len([d for d in os.listdir("earthquake") if os.path.isdir(f"earthquake/{d}")])
num_noise_samples_e = len([d for d in os.listdir("noise") if os.path.isdir(f"noise/{d}")])
num_noise_samples = num_eq_samples - num_noise_samples_e  # Balance EQ/Noise

print(f"\n---🔍 Bắt đầu tạo {num_noise_samples} sample noise ---")

for i in range(num_noise_samples):
    try:
        # === Tìm vùng noise hợp lệ
        for _ in range(50):  # Thử nhiều hơn để đỡ 204
            event = random.choice(catalog)
            origin_time = event.origins[0].time
            rand_offset = random.uniform(1200, 3600)  # 20–60 phút sau EQ
            rand_time = origin_time + rand_offset
            if is_interval_safe(rand_time):
                break
        else:
            print(f"⚠️ Không tìm được noise hợp lệ {i}")
            continue

        st = client.get_waveforms(
            network=NETWORK, station=STATION, location=LOCATION, channel=CHANNEL,
            starttime=rand_time,
            endtime=rand_time + 300  # 5 phút noise
        )
        st.detrend("demean")
        st.filter("bandpass", freqmin=1, freqmax=10)
        tr = st[0]

        fs = tr.stats.sampling_rate
        segment_length = 5
        segment_samples = int(segment_length * fs)

        # === Random length noise sample
        min_len = 0      
        max_len = 40      
        length = random.randint(min_len, max_len)

        frame_count = 0
        seq_dir = f"noise/{STATION}_sample_{i:03}"  # Giữ đồng bộ style với EQ

        for j in range(length):
            start_idx = j * segment_samples
            end_idx = start_idx + segment_samples
            if end_idx > len(tr.data):
                break

            # 🟢 Tạo folder chỉ khi có frame đầu
            if frame_count == 0:
                os.makedirs(seq_dir, exist_ok=True)

            segment = tr.data[start_idx:end_idx]
            frame_count += 1
            frame_name = f"{seq_dir}/frame_{frame_count:03}.png"
            ok = plot_frame_spectrogram_fixed(segment, fs, segment_length, frame_name)
            if not ok:
                continue  # ❌ Bỏ nếu frame quá yên tĩnh

        if frame_count == 0:
            print(f"⚠️ Noise {i} không đủ dữ liệu, skip.")
            continue

        print(f"✅ Noise {i} xong. Đã lưu {frame_count} frame ➜ {seq_dir}")

    except Exception as e:
        print(f"❌ Lỗi noise {i}: {e}")

