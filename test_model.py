import os
import re
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ==== Config ====
client = Client("IRIS")
INPUT_FILE = "obs_orientation_metadata_updated.txt"
OUTPUT_DIR = "ring_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_MAGNITUDE = 2.5
MAX_RADIUS = 2.0

# ==== Đọc file station list ====
stations = []
with open(INPUT_FILE, "r") as f:
    for line in f:
        parts = line.strip().split(", ")
        st = parts[0].split(": ")[1]
        start = parts[1].split(": ")[1]
        end = parts[2].split(": ")[1]
        lat = float(parts[3].split(": ")[1])
        lon = float(parts[4].split(": ")[1])
        stations.append((st, start, end, lat, lon))

# ==== Loop qua từng trạm ====
for STATION, start, end, lat, lon in stations:
    start_time = UTCDateTime(start)
    end_time = UTCDateTime(end)

    print(f"🔍 Checking picks for station {STATION} ({start} → {end})")

    catalog = client.get_events(starttime=start_time, endtime=end_time,
                                latitude=lat, longitude=lon,
                                maxradius=MAX_RADIUS, minmagnitude=MIN_MAGNITUDE)

    output_file = os.path.join(OUTPUT_DIR, f"ring_{STATION}.txt")
    with open(output_file, "w") as f:
        for ev in catalog:
            origin = ev.origins[0]
            f.write(f"Event: {origin.time}\n")
            f.write(f"  Hypocenter: lat={origin.latitude:.4f}, lon={origin.longitude:.4f}, depth={origin.depth/1000:.1f} km\n")

            # Arrivals
            if origin.arrivals:
                f.write("  Arrivals:\n")
                for arr in origin.arrivals:
                    f.write(f"    {arr.phase}\n")
            else:
                f.write("  Arrivals: None\n")

            # Picks (lọc đúng station)
            picks_for_station = [p for p in ev.picks if p.waveform_id.station_code == STATION]
            if picks_for_station:
                f.write("  Picks:\n")
                for pick in picks_for_station:
                    f.write(f"    {pick.phase_hint} @ {pick.time}\n")
            else:
                f.write("  Picks: None\n")

            f.write("\n")

    print(f"✅ Saved {output_file}")
