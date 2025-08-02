import os
import re
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace
from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
from obspy.taup import TauPyModel
from scipy.signal import spectrogram
from PIL import Image, ImageDraw

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
TARGET_SIZE = (512, 256)
os.makedirs("full_spectrogram", exist_ok=True)

MIN_MAGNITUDE = 3
MAX_RADIUS = 2.0

# TauP model và offset hiệu chỉnh
model = TauPyModel(model="iasp91")
P_OFFSET = 10 # dịch P-arrival sớm hơn 1.5 giây

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
        eq_lat = origin.latitude
        eq_lon = origin.longitude
        eq_depth = origin.depth / 1000.0  # km

        try:
            # ==== Tính P-arrival bằng TauP + offset ====
            dist_m, az, baz = gps2dist_azimuth(lat_sta, lon_sta, eq_lat, eq_lon)
            dist_deg = kilometers2degrees(dist_m / 1000.0)
            arrivals = model.get_travel_times(source_depth_in_km=eq_depth,
                                              distance_in_degree=dist_deg,
                                              phase_list=["P"])
            if arrivals:
                p_travel = arrivals[0].time
                p_arrival = origin_time + p_travel - P_OFFSET
            else:
                p_arrival = origin_time

            print(f"Event {i}: Origin={origin_time}, P-arrival={p_arrival} (Offset {P_OFFSET}s)")

            # ==== Load waveform ====
            st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                      starttime=origin_time - 60,
                                      endtime=origin_time + 420)

            st.detrend("demean")
            st.filter("bandpass", freqmin=1, freqmax=10)
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

            # ==== Spectrogram ====
            specZ, t_axis = compute_spec(trZ, fs)
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

            # ==== Vẽ P-arrival ====
            window_start = origin_time - 60
            offset_sec = p_arrival - window_start
            total_duration = 480.0
            p_x = int((offset_sec / total_duration) * TARGET_SIZE[0])

            # RGB
            img_rgb = Image.fromarray(spec_resized)
            draw_rgb = ImageDraw.Draw(img_rgb)
            draw_rgb.line([(p_x, 0), (p_x, TARGET_SIZE[1])], fill=(0, 255, 0), width=2)
            img_rgb.save(f"full_spectrogram/{STATION}_eq_{i:03}_CRED_RGB_P.png")

            # Từng kênh
            for idx, ch in enumerate(["BHZ", "BHN", "BHE"]):
                img_ch = Image.fromarray(spec_resized[:, :, idx])
                draw_ch = ImageDraw.Draw(img_ch)
                draw_ch.line([(p_x, 0), (p_x, TARGET_SIZE[1])], fill=255, width=2)
                img_ch.save(f"full_spectrogram/{STATION}_eq_{i:03}_{ch}_P.png")

            print(f"✅ Saved {STATION} event {i} spectrogram with P-arrival line")

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
