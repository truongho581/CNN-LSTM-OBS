import os
import re
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, Trace, Stream
from obspy.geodetics import locations2degrees
from scipy.signal import spectrogram
from seisbench.models.pickblue import PickBlue
import matplotlib.pyplot as plt

# ==== Config ====
client = Client("IRIS")
model = PickBlue(base="eqtransformer")

NETWORK = "7D"
LOCATION = "--"
CHANNEL = "BH?"
MIN_MAGNITUDE = 2.5
MIN_RADIUS = 0
MAX_RADIUS = 2.0

SAVE_DIR = "spectrogram_regional"
os.makedirs(SAVE_DIR, exist_ok=True)

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
        bh1_angle = float(re.sub(r"[^\d\.\-]", "", bh1_str) or 0.0)
        if -1500 < depth < -500:
            stations.append((st, UTCDateTime(start), UTCDateTime(end), lat, lon, bh1_angle))

# ==== Main loop ====
for STATION, start_time, end_time, lat_sta, lon_sta, bh1_angle in stations:
    print(f"\n📡 {STATION} ({start_time.date} → {end_time.date})")

    try:
        catalog = client.get_events(
            starttime=start_time,
            endtime=end_time,
            latitude=lat_sta,
            longitude=lon_sta,
            minradius=MIN_RADIUS,
            maxradius=MAX_RADIUS,
            minmagnitude=MIN_MAGNITUDE
        )
    except Exception as e:
        print(f"❌ Lỗi tải catalog: {e}")
        continue

    catalog.events.sort(key=lambda e: e.origins[0].time)

    for i, event in enumerate(catalog):
        origin = event.origins[0]
        origin_time = origin.time
        magnitude = event.magnitudes[0].mag
        dist_deg = locations2degrees(origin.latitude, origin.longitude, lat_sta, lon_sta)

        try:
            # ==== Tải waveform đủ 3 kênh ====
            st = client.get_waveforms(
                network=NETWORK,
                station=STATION,
                location=LOCATION,
                channel=CHANNEL,
                starttime=origin_time - 30,
                endtime=origin_time + 330
            )
            st.detrend("demean")
            st.filter("bandpass", freqmin=2, freqmax=10)

            if not all(st.select(channel=f"BH{c}") for c in ["Z", "1", "2"]):
                print(f"⚠️ {STATION} event {i}: thiếu BHZ/BH1/BH2 → bỏ qua")
                continue

            # ==== Xoay BH1/BH2 thành BHN/BHE ====
            phi_rad = np.deg2rad(bh1_angle)
            tr1 = st.select(channel="BH1")[0]
            tr2 = st.select(channel="BH2")[0]
            trZ = st.select(channel="BHZ")[0]

            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)

            trN = Trace(data=north, header=tr1.stats); trN.stats.channel = "BHN"
            trE = Trace(data=east, header=tr2.stats);  trE.stats.channel = "BHE"
            stream = Stream([trZ, trN, trE])
            fs = trZ.stats.sampling_rate
            trace_start = trZ.stats.starttime

            # ==== Pick P/S ====
            result = model.classify(stream)
            parsed_picks = []
            for pick in result.picks:
                s = str(pick)
                match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", s)
                if match:
                    pick_time = UTCDateTime(match.group(0))
                    phase = "P" if s.strip().endswith("P") else "S" if s.strip().endswith("S") else "?"
                    parsed_picks.append((pick_time, phase))

            if not parsed_picks:
                print(f"⚠️ {STATION} event {i}: PickBlue không pick được pha nào → bỏ qua")
                continue

            # ==== Tạo spectrogram (BHZ) ====
            f, t_spec, Sxx = spectrogram(trZ.data, fs=fs, nperseg=256, noverlap=128)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            # ==== Vẽ + overlay ====
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap="viridis",vmin = -60, vmax=60 )
            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Time (s)")
            plt.colorbar(label="Power (dB)")
            plt.axvline(x=30, color="white", linestyle="--", linewidth=2, label="Origin Time")


            drawn_labels = set()
            for pick_time, phase in parsed_picks:
                t_offset = pick_time - trace_start
                if 0 <= t_offset <= trZ.stats.npts / fs:
                    color = "cyan" if phase == "P" else "lime"
                    label = phase if phase not in drawn_labels else None
                    plt.axvline(x=t_offset, color=color, linestyle="--", linewidth=2, label=label)
                    drawn_labels.add(phase)

            plt.legend()
            plt.tight_layout()

            # ==== Lưu PNG ====
            fname = f"{STATION}_eq_{i:03}_M{magnitude:.1f}_dist{dist_deg:.1f}.png"
            out_path = os.path.join(SAVE_DIR, fname)
            plt.savefig(out_path, dpi=300)
            plt.close()

            print(f"✅ {STATION} event {i}: saved {fname}")

        except Exception as e:
            print(f"❌ Lỗi xử lý {STATION} event {i}: {e}")
            continue
