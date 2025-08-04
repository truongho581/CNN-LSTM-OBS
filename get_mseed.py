import os
import re
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Trace, Stream

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
LOCATION = "--"
output_dir = "mseed_data"
os.makedirs(output_dir, exist_ok=True)

MIN_MAGNITUDE = 3
MAX_RADIUS = 2

# ==== Load stations ====
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

    station_dir = os.path.join(output_dir, STATION)
    os.makedirs(station_dir, exist_ok=True)

    catalog = client.get_events(starttime=start_time, endtime=end_time,
                                latitude=lat_sta, longitude=lon_sta,
                                maxradius=MAX_RADIUS, minmagnitude=MIN_MAGNITUDE)
    print(f"📡 {STATION}: {len(catalog)} events")

    events_sorted = sorted(catalog, key=lambda e: e.origins[0].time)
    event_times = [ev.origins[0].time for ev in events_sorted]
    for i, event in enumerate(events_sorted):
        origin = event.origins[0]
        origin_time = origin.time

        try:
            start_cut = origin_time - 30
            end_cut = origin_time + 120

            st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                      starttime=start_cut, endtime=end_cut)
            if len(st) == 0:
                continue

            st.detrend("demean")
            phi_rad = np.deg2rad(bh1_angle)

            tr1 = st.select(channel="BH1")[0]
            tr2 = st.select(channel="BH2")[0]
            trZ = st.select(channel="BHZ")[0]

            north = tr1.data * np.cos(phi_rad) + tr2.data * np.sin(phi_rad)
            east  = -tr1.data * np.sin(phi_rad) + tr2.data * np.cos(phi_rad)

            trN = Trace(data=north, header=tr1.stats)
            trE = Trace(data=east, header=tr2.stats)
            trZ.stats.channel = "BHZ"
            trN.stats.channel = "BHN"
            trE.stats.channel = "BHE"

            st_out = Stream([trZ, trN, trE])

            mseed_path = os.path.join(station_dir, f"{STATION}_eq_{i:03}.mseed")
            st_out.write(mseed_path, format="MSEED")
            print(f"✅ Saved {mseed_path}")

        except Exception as e:
            print(f"❌ Lỗi {STATION} event {i}: {e}")
            continue
