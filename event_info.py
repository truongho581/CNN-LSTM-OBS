from OBS_station import start_time, end_time, client, LOCATION, NETWORK, CHANNEL, station_data
from obspy.geodetics import locations2degrees

# ==== Cấu hình ====
STATION, lat, lon, elev = station_data
radius = 0       # bán kính tối thiểu (~220 km)
max_radius = 2   # bán kính tối đa (~1110 km)
minmagni = 2.5     # độ lớn tối thiểu

# ==== Truy vấn catalog sự kiện ====
catalog = client.get_events(
    starttime=start_time,
    endtime=end_time,
    latitude=lat,
    longitude=lon,
    minradius=radius,
    maxradius=max_radius,
    minmagnitude=minmagni
)

# ==== Sắp xếp theo thời gian ====
catalog.events.sort(key=lambda e: e.origins[0].time)

# ==== In kết quả ====
if __name__ == "__main__":
    print(f"---🔍 Tổng cộng {len(catalog)} sự kiện---\n")

    for i, event in enumerate(catalog):
        origin = event.origins[0]
        origin_time = origin.time
        magnitude = event.magnitudes[0].mag
        lat_eq = origin.latitude
        lon_eq = origin.longitude

        dist_deg = locations2degrees(lat, lon, lat_eq, lon_eq)

        print(f"{i:02d}. ⏰ {origin_time} | M = {magnitude:.1f} | Dist = {dist_deg:.2f}°")
