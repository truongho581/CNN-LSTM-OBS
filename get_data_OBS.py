import os
import re
import numpy as np
import random
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

MIN_MAGNITUDE = 3
MAX_RADIUS = 1.5

# ==== TauP model ====
taup_model = TauPyModel(model="iasp91")

# ==== Hàm hiệu chỉnh Cascadia tuyến tính ====
def cascadia_correction(p_theo, origin_time, dist_deg):
    """
    Hiệu chỉnh tuyến tính dựa vào khoảng cách event–station.
    - 0.0°: offset = 0s
    - 1.0°: offset = -12s
    Không bao giờ lùi qua origin_time.
    """
    a = (-12.0 - 0.0) / (1.0 - 0.0)   # slope để 1.0° = -12s
    b = 0.0
    offset = a * dist_deg + b

    # Giới hạn offset tối đa
    offset = max(offset, -20.0)

    p_corr = p_theo + offset
    if p_corr < origin_time:
        p_corr = origin_time
    return p_corr

# ==== Đọc metadata ====
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

# ==== Pad ====
def pad_or_trim(data, expected_samples):
    if len(data) < expected_samples:
        pad = np.zeros(expected_samples - len(data))
        return np.concatenate([data, pad])
    elif len(data) > expected_samples:
        return data[:expected_samples]
    return data

# ==== Noise sample ====
def create_noise_sample(station_dir, STATION, index, ref_time, fs, expected_samples, phi_rad, event_times):
    try:
        for _ in range(10):
            offset_min = random.randint(1200, 2400)  # 20–40 phút
            direction = random.choice([-1, 1])
            noise_start = ref_time + direction * offset_min
            noise_end = noise_start + 100

            # tránh trùng event
            if any((noise_start <= ev <= noise_end) or (ev <= noise_start <= ev + 60) for ev in event_times):
                continue

            st_noise = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?", 
                                            starttime=noise_start, endtime=noise_end)
            st_noise.detrend("demean")
            st_noise.filter("bandpass", freqmin=2, freqmax=8)
            for tr in st_noise:
                tr.data = tr.data.astype(np.float64)
                std = np.std(tr.data)
                if std > 0:
                    tr.data /= std

            tr1 = st_noise.select(channel="BH1")[0]
            tr2 = st_noise.select(channel="BH2")[0]
            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)
            trZ = st_noise.select(channel="BHZ")[0]

            Z = pad_or_trim(trZ.data, expected_samples)
            N = pad_or_trim(north, expected_samples)
            E = pad_or_trim(east, expected_samples)

            tensor = np.stack([Z, N, E], axis=0)

            noise_dir = os.path.join(station_dir, f"noise_{index:03}")
            os.makedirs(noise_dir, exist_ok=True)
            np.savez(os.path.join(noise_dir, "sample.npz"), waveform=tensor, fs=fs)
            with open(os.path.join(noise_dir, "label.txt"), "w") as lf:
                lf.write(" ".join(["0"]*20))

            print(f"🎵 Saved NOISE {noise_dir}")
            return
        print(f"⚠️ Không tìm được khoảng noise phù hợp cho {STATION}")
    except Exception as e:
        print(f"❌ Noise sample error {STATION}: {e}")

# ==== Loop ====
for STATION, start, end, lat_sta, lon_sta, bh1_angle in stations:
    start_time = UTCDateTime(start)
    end_time = UTCDateTime(end)
    phi_rad = np.deg2rad(bh1_angle)

    station_dir = os.path.join(base_dir, STATION)
    os.makedirs(station_dir, exist_ok=True)

    catalog = client.get_events(starttime=start_time, endtime=end_time,
                                latitude=lat_sta, longitude=lon_sta,
                                maxradius=MAX_RADIUS, minmagnitude=MIN_MAGNITUDE)
    event_times = [ev.origins[0].time for ev in catalog]

    for i, event in enumerate(catalog):
        origin = event.origins[0]
        origin_time = origin.time

        try:
            # ==== Tính P-arrival bằng TauP + hiệu chỉnh tuyến tính ====
            ev_lat = event.origins[0].latitude
            ev_lon = event.origins[0].longitude
            ev_depth_km = event.origins[0].depth / 1000.0

            dist_deg = locations2degrees(ev_lat, ev_lon, lat_sta, lon_sta)
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=ev_depth_km,
                distance_in_degree=dist_deg,
                phase_list=["P", "p", "Pn", "Pg"]
            )

            if arrivals:
                p_theo = origin_time + arrivals[0].time
                p_arrival = cascadia_correction(p_theo, origin_time, dist_deg)
            else:
                p_arrival = origin_time + 5.0
                print(f"⚠️ {STATION} Event {i}: TauP không có P, fallback={p_arrival}")

            # ==== Tải waveform ====
            st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                      starttime=origin_time - 60,
                                      endtime=origin_time + 420)
            st.detrend("demean")
            st.filter("bandpass", freqmin=2, freqmax=8)
            for tr in st:
                tr.data = tr.data.astype(np.float64)
                std = np.std(tr.data)
                if std > 0:
                    tr.data /= std

            trZ = st.select(channel="BHZ")[0]
            tr1 = st.select(channel="BH1")[0]
            tr2 = st.select(channel="BH2")[0]

            # ==== Xoay N/E ====
            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)
            trN = Trace(data=north, header=tr1.stats)
            trE = Trace(data=east, header=tr2.stats)

            # ==== Cắt 100s từ p_arrival ====
            fs = trZ.stats.sampling_rate
            frame_len = 5
            total_len = 100
            n_noise = random.randint(1, 5)
            start_cut = p_arrival - n_noise * frame_len
            end_cut = start_cut + total_len

            trZ_cut = trZ.copy().trim(starttime=start_cut, endtime=end_cut)
            trN_cut = trN.copy().trim(starttime=start_cut, endtime=end_cut)
            trE_cut = trE.copy().trim(starttime=start_cut, endtime=end_cut)

            expected_samples = int(total_len * fs)
            Z = pad_or_trim(trZ_cut.data, expected_samples)
            N = pad_or_trim(trN_cut.data, expected_samples)
            E = pad_or_trim(trE_cut.data, expected_samples)

            tensor = np.stack([Z, N, E], axis=0)

            eq_dir = os.path.join(station_dir, f"eq_{i:03}")
            os.makedirs(eq_dir, exist_ok=True)
            np.savez(os.path.join(eq_dir, "sample.npz"),
                     waveform=tensor,
                     fs=fs,
                     p_arrival=float(p_arrival.timestamp))

            labels = [0]*n_noise + [1]*(20-n_noise)
            with open(os.path.join(eq_dir, "label.txt"), "w") as lf:
                lf.write(" ".join(map(str, labels)))

            print(f"✅ Saved EQ {eq_dir} (p_arrival={p_arrival}, dist={dist_deg:.2f}°)")

            # ==== Tạo noise sample ====
            create_noise_sample(station_dir, STATION, i, p_arrival, fs, expected_samples, phi_rad, event_times)

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
