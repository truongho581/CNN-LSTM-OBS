import os
import re
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace, Stream
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from scipy.signal import spectrogram
from PIL import Image, ImageDraw
from seisbench.models.pickblue import PickBlue

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
TARGET_SIZE = (512, 256)
os.makedirs("full_spectrogram", exist_ok=True)

MIN_MAGNITUDE = 3
MAX_RADIUS = 2

taup_model = TauPyModel(model="iasp91")
model = PickBlue(base="eqtransformer")  # Load PickBlue

# ==== Hàm hiệu chỉnh Cascadia tuyến tính ====
def cascadia_correction(p_theo, origin_time, dist_deg):
    if dist_deg <= 1.0:
        a = (-12.0 - 0.0) / 1.0
        offset = a * dist_deg
    else:
        a = (-12.3 - -12.0) / (1.5 - 1.0)
        offset = -12.0 + a * (dist_deg - 1.0)
    return max(p_theo + offset, origin_time)

# ==== Hàm tạo spectrogram RGB đã clip dB cố định ====
def make_spectrogram(data, fs, fmin=2, fmax=10, clip_min=-80, clip_max=0, gamma=1):
    nperseg = int(0.2 * fs)
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f_mask = (f >= fmin) & (f <= fmax)
    Sxx = Sxx[f_mask, :]
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    Sxx_db = np.clip(Sxx_db, clip_min, clip_max)
    spec = (Sxx_db - clip_min) / (clip_max - clip_min)
    spec = np.power(spec, gamma)
    spec = spec[::-1, :]

    img = Image.fromarray((spec * 255).astype(np.uint8)).convert("L")
    img = img.resize(TARGET_SIZE, Image.BICUBIC)
    return np.array(img, dtype=np.float32) / 255.0

# ==== Load station metadata ====
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
        bh1 = float(re.sub(r"[^\d\.\-]", "", bh1_str) or 0.0)
        if depth < -500:
            stations.append((st, start, end, lat, lon, bh1))

# ==== Main loop ====
for STATION, start, end, lat_sta, lon_sta, bh1_angle in stations:
    start_time = UTCDateTime(start)
    end_time = UTCDateTime(end)
    phi_rad = np.deg2rad(bh1_angle)

    print(f"\n📡 {STATION} ({start} → {end}) BH1={bh1_angle}°")

    catalog = client.get_events(starttime=start_time, endtime=end_time,
                                latitude=lat_sta, longitude=lon_sta,
                                maxradius=MAX_RADIUS, minmagnitude=MIN_MAGNITUDE)
    events_sorted = sorted(catalog, key=lambda e: e.origins[0].time)

    for i, event in enumerate(events_sorted):
        origin = event.origins[0]
        origin_time = origin.time

        try:
            dist_deg = locations2degrees(origin.latitude, origin.longitude, lat_sta, lon_sta)
            arrivals = taup_model.get_travel_times(
                source_depth_in_km=origin.depth / 1000.0,
                distance_in_degree=dist_deg,
                phase_list=["P", "p", "Pn", "Pg"]
            )
            if arrivals:
                p_theo = origin_time + arrivals[0].time
                p_arrival = cascadia_correction(p_theo, origin_time, dist_deg)
            else:
                p_arrival = origin_time + 5.0

            st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                      starttime=origin_time - 30,
                                      endtime=origin_time + 150)
            st.detrend("demean")
            st.filter("bandpass", freqmin=2, freqmax=8)
            for tr in st:
                tr.data = tr.data.astype(np.float64)
                std = np.std(tr.data)
                if std > 0:
                    tr.data /= std

            tr1 = st.select(channel="BH1")[0]
            tr2 = st.select(channel="BH2")[0]
            trZ = st.select(channel="BHZ")[0]
            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)
            trN = Trace(data=north, header=tr1.stats); trN.stats.channel = "BHN"
            trE = Trace(data=east, header=tr2.stats);  trE.stats.channel = "BHE"
            fs = trZ.stats.sampling_rate

            stream = Stream([trZ, trN, trE])
            result = model.classify(stream)

            p_pick_time, s_pick_time = None, None
            for pick in result.picks:
                pick_str = str(pick).strip()
                if pick_str.endswith("P") and not p_pick_time:
                    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", pick_str)
                    if match:
                        p_pick_time = UTCDateTime(match.group(0))
                elif pick_str.endswith("S") and not s_pick_time:
                    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", pick_str)
                    if match:
                        s_pick_time = UTCDateTime(match.group(0))

            # === RGB spectrogram từ Z, N, E với dB clip ===
            specZ = make_spectrogram(trZ.data, fs)
            specN = make_spectrogram(trN.data, fs)
            specE = make_spectrogram(trE.data, fs)
            rgb_array = np.stack([specZ, specN, specE], axis=-1)
            rgb_img = (rgb_array * 255).astype(np.uint8)
            img = Image.fromarray(rgb_img)

            # === Overlay line các pha ===
            draw = ImageDraw.Draw(img)
            total_duration = 180
            window_start = origin_time - 30
            def time_to_x(t): return int(((t - window_start) / total_duration) * TARGET_SIZE[0])

            draw.line([(time_to_x(origin_time), 0), (time_to_x(origin_time), TARGET_SIZE[1])], fill=(255, 0, 0), width=2)
            draw.line([(time_to_x(p_arrival), 0), (time_to_x(p_arrival), TARGET_SIZE[1])], fill=(0, 255, 0), width=2)

            if p_pick_time:
                draw.line([(time_to_x(p_pick_time), 0), (time_to_x(p_pick_time), TARGET_SIZE[1])], fill=(0, 255, 255), width=2)
            if s_pick_time:
                draw.line([(time_to_x(s_pick_time), 0), (time_to_x(s_pick_time), TARGET_SIZE[1])], fill=(255, 255, 0), width=2)

            fname = f"full_spectrogram/{STATION}_eq_{i:03}_TauP_PickBlue.png"
            img.save(fname)
            print(f"✅ {STATION} event {i}: saved {fname}")

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
