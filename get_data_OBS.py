import os
import re
import numpy as np
import random
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from scipy.signal import spectrogram
from PIL import Image

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

MIN_MAGNITUDE = 3
MAX_RADIUS = 2
TARGET_SIZE = (64, 64)  # mỗi frame spectrogram 64×64

# ==== TauP model ====
taup_model = TauPyModel(model="iasp91")

# ==== Cascadia correction ====
def cascadia_correction(p_theo, origin_time, dist_deg):
    if dist_deg <= 1.0:
        a = (-12.0 - 0.0) / 1.0
        offset = a * dist_deg
    else:
        a = (-12.3 - -12.0) / (1.5 - 1.0)
        offset = -12.0 + a * (dist_deg - 1.0)
    p_corr = p_theo + offset
    if p_corr < origin_time:
        p_corr = origin_time
    return p_corr

# ==== Tính spectrogram và resize 64x64 ====
def make_spectrogram(trace, fs, fmin=2, fmax=10):
    nperseg = int(1.0 * fs)    # dùng 1s để có độ phân giải thời gian cao
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_mask = (f >= fmin) & (f <= fmax)
    Sxx = Sxx[f_mask, :]
    Sxx_log = np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log) + 1e-8)
    img = Image.fromarray((Sxx_norm * 255).astype(np.uint8))
    img = img.resize(TARGET_SIZE, Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0  # (64,64)

# ==== Pad / Trim ====
def pad_or_trim(data, expected_samples):
    if len(data) < expected_samples:
        pad = np.zeros(expected_samples - len(data))
        return np.concatenate([data, pad])
    elif len(data) > expected_samples:
        return data[:expected_samples]
    return data

# ==== Chia waveform 100s thành 20 frame 5s ====
def split_into_frames(data, fs, frame_len=5):
    samples_per_frame = int(frame_len * fs)
    frames = []
    for i in range(20):
        start = i * samples_per_frame
        end = start + samples_per_frame
        frames.append(data[start:end])
    return frames

# ==== Tạo sample noise ====
def create_noise_sample(station_dir, STATION, index, ref_time, fs, phi_rad, event_times):
    try:
        for _ in range(10):
            offset_min = random.randint(1200, 2400)
            direction = random.choice([-1, 1])
            noise_start = ref_time + direction * offset_min
            noise_end = noise_start + 100

            if any((noise_start <= ev <= noise_end) or (ev <= noise_start <= ev + 60) for ev in event_times):
                continue

            st_noise = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                            starttime=noise_start, endtime=noise_end)
            if len(st_noise) == 0:
                continue

            st_noise.detrend("demean")
            st_noise.filter("bandpass", freqmin=2, freqmax=8)
            for tr in st_noise:
                tr.data = tr.data.astype(np.float64)
                std = np.std(tr.data)
                if std > 0:
                    tr.data /= std

            tr1 = st_noise.select(channel="BH1")[0]
            tr2 = st_noise.select(channel="BH2")[0]
            trZ = st_noise.select(channel="BHZ")[0]

            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)

            # ==== Chia thành 20 frame ====
            Z_frames = split_into_frames(trZ.data, fs)
            N_frames = split_into_frames(north, fs)
            E_frames = split_into_frames(east, fs)

            spec_frames = []
            for z, n, e in zip(Z_frames, N_frames, E_frames):
                specZ = make_spectrogram(z, fs)
                specN = make_spectrogram(n, fs)
                specE = make_spectrogram(e, fs)
                spec_rgb = np.stack([specZ, specN, specE], axis=-1)  # (64,64,3)
                spec_frames.append(spec_rgb)

            spec_seq = np.stack(spec_frames, axis=0)  # (20,64,64,3)

            noise_dir = os.path.join(station_dir, f"noise_{index:03}")
            os.makedirs(noise_dir, exist_ok=True)
            np.savez(os.path.join(noise_dir, "sample.npz"), spec=spec_seq, fs=fs)
            with open(os.path.join(noise_dir, "label.txt"), "w") as lf:
                lf.write(" ".join(["0"]*20))

            print(f"🎵 Saved NOISE {noise_dir} spectrogram sequence")
            return
        print(f"⚠️ Không tìm được noise phù hợp cho {STATION}")
    except Exception as e:
        print(f"❌ Noise sample error {STATION}: {e}")

# ==== Loop qua các trạm ====
stations = []
with open("obs_orientation_metadata_updated.txt", "r") as f:
    for line in f:
        parts = line.strip().split(", ")
        st = parts[0].split(": ")[1]
        start = parts[1].split(": ")[1]
        end = parts[2].split(": ")[1]
        lat = float(parts[3].split(": ")[1])
        lon = float(parts[4].split(": ")[1])
        depth = float(parts[5].split(": ")[1])
        bh1_str = parts[6].split(": ")[1].strip()
        bh1_clean = re.sub(r"[^0-9.\-]", "", bh1_str)
        bh1 = float(bh1_clean) if bh1_clean else 0.0

        if depth < -500:
            stations.append((st, start, end, lat, lon, bh1))

print(f"📌 Loaded {len(stations)} stations")

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
    print(f"📡 {STATION}: {len(catalog)} events")

    for i, event in enumerate(catalog):
        origin = event.origins[0]
        origin_time = origin.time

        try:
            ev_lat = origin.latitude
            ev_lon = origin.longitude
            ev_depth_km = origin.depth / 1000.0
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

            st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                      starttime=origin_time - 60,
                                      endtime=origin_time + 420)
            if len(st) == 0:
                continue

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

            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)

            fs = trZ.stats.sampling_rate
            if fs is None:
                continue

            frame_len = 5
            total_len = 100
            n_noise = random.randint(1, 5)
            start_cut = p_arrival - n_noise * frame_len
            end_cut = start_cut + total_len

            trZ_cut = trZ.copy().trim(starttime=start_cut, endtime=end_cut)
            trN_cut = Trace(data=north, header=tr1.stats).trim(starttime=start_cut, endtime=end_cut)
            trE_cut = Trace(data=east, header=tr2.stats).trim(starttime=start_cut, endtime=end_cut)

            Z_frames = split_into_frames(trZ_cut.data, fs)
            N_frames = split_into_frames(trN_cut.data, fs)
            E_frames = split_into_frames(trE_cut.data, fs)

            spec_frames = []
            for z, n, e in zip(Z_frames, N_frames, E_frames):
                specZ = make_spectrogram(z, fs)
                specN = make_spectrogram(n, fs)
                specE = make_spectrogram(e, fs)
                spec_rgb = np.stack([specZ, specN, specE], axis=-1)  # (64,64,3)
                spec_frames.append(spec_rgb)

            spec_seq = np.stack(spec_frames, axis=0)  # (20,64,64,3)

            eq_dir = os.path.join(station_dir, f"eq_{i:03}")
            os.makedirs(eq_dir, exist_ok=True)
            np.savez(os.path.join(eq_dir, "sample.npz"),
                     spec=spec_seq,
                     fs=fs,
                     p_arrival=float(p_arrival.timestamp))

            labels = [0]*n_noise + [1]*(20-n_noise)
            with open(os.path.join(eq_dir, "label.txt"), "w") as lf:
                lf.write(" ".join(map(str, labels)))

            print(f"✅ Saved EQ {eq_dir} spectrogram sequence (20x64x64x3)")

            create_noise_sample(station_dir, STATION, i, p_arrival, fs, phi_rad, event_times)

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
