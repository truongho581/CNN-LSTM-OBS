import os
import re
import numpy as np
import random
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
base_dir = "data"
os.makedirs(base_dir, exist_ok=True)

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

# ==== Hàm pad/truncate ====
def pad_or_trim(data, expected_samples):
    if len(data) < expected_samples:
        pad = np.zeros(expected_samples - len(data))
        return np.concatenate([data, pad])
    elif len(data) > expected_samples:
        return data[:expected_samples]
    return data

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
            st.filter("bandpass", freqmin=2, freqmax=8)
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

            # ==== Tìm P-onset bằng STA/LTA ====
            tr_cut = trZ.copy().trim(starttime=origin_time, endtime=origin_time + 60)
            sta_win = int(0.25 * fs)    
            lta_win = int(10 * fs)     
            cft = classic_sta_lta(tr_cut.data, sta_win, lta_win)
            trigs = trigger_onset(cft, 2.2, 1.0)

            if len(trigs) > 0:
                onset_sample = trigs[0][0]
                p_arrival = tr_cut.stats.starttime + onset_sample / fs
            else:
                p_arrival = origin_time

            # ==== Cắt cứng 100s với random noise frame ====
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

            tensor = np.stack([Z, N, E], axis=0)  # shape (3, samples)

            # ==== Tạo thư mục sample ====
            sample_dir = os.path.join(base_dir, f"{STATION}_eq_{i:03}")
            os.makedirs(sample_dir, exist_ok=True)

            # ==== Lưu file .npz ====
            np.savez(os.path.join(sample_dir, "sample.npz"),
                     waveform=tensor,
                     fs=fs,
                     p_arrival=float(p_arrival.timestamp),
                     n_noise=n_noise)

            # ==== Lưu label.txt ====
            labels = [0]*n_noise + [1]*(20-n_noise)
            with open(os.path.join(sample_dir, "label.txt"), "w") as lf:
                lf.write(" ".join(map(str, labels)))

            print(f"✅ Saved {sample_dir}: tensor {tensor.shape}, noise={n_noise}")

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
