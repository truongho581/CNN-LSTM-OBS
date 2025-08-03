import os
import re
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace
from scipy.signal import spectrogram
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from PIL import Image, ImageDraw

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
TARGET_SIZE = (512, 256)
os.makedirs("full_spectrogram", exist_ok=True)

MIN_MAGNITUDE = 3
MAX_RADIUS = 1.5

# ==== Đọc file station metadata ====
stations = []
with open("obs_orientation_metadata_updated.txt", "r") as f:
    for line in f:
        parts = line.strip().split(", ")
        st = parts[0].split(": ")[1]
        start = parts[1].split(": ")[1]
        end = parts[2].split(": ")[1]
        lat = float(parts[3].split(": ")[1])
        lon = float(parts[4].split(": ")[1])
        bh1_str = parts[6].split(": ")[1].strip()
        bh1_clean = re.sub(r"[^0-9.\-]", "", bh1_str)
        bh1 = float(bh1_clean) if bh1_clean else 0.0
        stations.append((st, start, end, lat, lon, bh1))

# ==== Hàm tạo spectrogram ====
def compute_spec(tr, fs):
    nperseg = int(5.12 * fs)
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(tr.data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)
    f_mask = (f >= 2) & (f <= 10)
    return power_db[f_mask, :], t

# ==== Loop ====
for STATION, start, end, lat_sta, lon_sta, bh1_angle in stations:
    start_time = UTCDateTime(start)
    end_time = UTCDateTime(end)
    phi_rad = np.deg2rad(bh1_angle)

    print(f"\n📡 Processing {STATION} ({start} → {end}), BH1={bh1_angle}°")

    catalog = client.get_events(starttime=start_time, endtime=end_time,
                                latitude=lat_sta, longitude=lon_sta,
                                maxradius=MAX_RADIUS, minmagnitude=MIN_MAGNITUDE)

    for i, event in enumerate(catalog):
        origin = event.origins[0]
        origin_time = origin.time

        try:
            # ==== Load waveform ====
            st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                      starttime=origin_time - 60,
                                      endtime=origin_time + 420)

            st.detrend("demean")
            st.filter("bandpass", freqmin=2, freqmax=8)  # filter hẹp hơn để P rõ hơn
            for tr in st:
                tr.data = tr.data.astype(np.float64)
                std = np.std(tr.data)
                if std > 0:
                    tr.data /= std

            # ==== Xoay BH1/BH2 sang N/E ====
            tr1 = st.select(channel="BH1")[0]
            tr2 = st.select(channel="BH2")[0]
            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)

            trN = Trace(data=north, header=tr1.stats)
            trE = Trace(data=east, header=tr2.stats)
            trN.stats.channel = "BHN"
            trE.stats.channel = "BHE"
            trZ = st.select(channel="BHZ")[0]

            fs = trZ.stats.sampling_rate

            # ==== Lấy 60 giây sau origin để tìm P bằng STA/LTA ====
            tr_cut = trZ.copy().trim(starttime=origin_time, endtime=origin_time + 60)

            sta_win = int(0.25 * fs)    # 0.5 giây ~ 25 mẫu @ 50Hz
            lta_win = int(10 * fs)     # 10 giây ~ 500 mẫu
            cft = classic_sta_lta(tr_cut.data, sta_win, lta_win)
            trigs = trigger_onset(cft, 2.2, 1.0)

            if len(trigs) > 0:
                onset_sample = trigs[0][0]
                p_arrival = tr_cut.stats.starttime + onset_sample / fs
                print(f"Event {i}: P-onset STA/LTA = {p_arrival}")
            else:
                p_arrival = origin_time
                print(f"Event {i}: Không thấy trigger, dùng Origin = {p_arrival}")

            # ==== Spectrogram ====
            specZ, _ = compute_spec(trZ, fs)
            specN, _ = compute_spec(trN, fs)
            specE, _ = compute_spec(trE, fs)
            spec_stack = np.stack([specZ, specN, specE], axis=-1)

            # Z-score & scale
            mean = np.mean(spec_stack)
            std = np.std(spec_stack) + 1e-8
            spec_stack = (spec_stack - mean) / std
            min_val, max_val = np.min(spec_stack), np.max(spec_stack)
            spec_stack = (spec_stack - min_val) / (max_val - min_val + 1e-8)
            spec_stack = (spec_stack * 255).astype(np.uint8)

            # Resize
            spec_stack = spec_stack[::-1, :, :]
            spec_resized = np.array(Image.fromarray(spec_stack).resize(TARGET_SIZE))

            # ==== Vẽ Origin (đỏ) và P-onset (xanh) ====
            window_start = origin_time - 60
            total_duration = 480.0

            # Origin line
            offset_origin = origin_time - window_start
            o_x = int((offset_origin / total_duration) * TARGET_SIZE[0])

            # P-onset line
            offset_p = p_arrival - window_start
            p_x = int((offset_p / total_duration) * TARGET_SIZE[0])

            # Vẽ trên ảnh RGB
            img_rgb = Image.fromarray(spec_resized)
            draw_rgb = ImageDraw.Draw(img_rgb)
            draw_rgb.line([(o_x, 0), (o_x, TARGET_SIZE[1])], fill=(255, 0, 0), width=2)  # đỏ = Origin
            draw_rgb.line([(p_x, 0), (p_x, TARGET_SIZE[1])], fill=(0, 255, 0), width=2)  # xanh = P-onset
            img_rgb.save(f"full_spectrogram/{STATION}_eq_{i:03}_CRED_RGB_P.png")

            # Vẽ trên từng kênh riêng
            for idx, ch in enumerate(["BHZ", "BHN", "BHE"]):
                img_ch = Image.fromarray(spec_resized[:, :, idx])
                draw_ch = ImageDraw.Draw(img_ch)
                draw_ch.line([(o_x, 0), (o_x, TARGET_SIZE[1])], fill=128, width=2)  # Origin xám
                draw_ch.line([(p_x, 0), (p_x, TARGET_SIZE[1])], fill=255, width=2)  # P-onset trắng
                img_ch.save(f"full_spectrogram/{STATION}_eq_{i:03}_{ch}_P.png")

            print(f"✅ Saved {STATION} event {i} spectrogram (Origin+P-onset lines)")

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
