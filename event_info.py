# Lấy thông tin về thời gian, độ lớn của sự kiện 
from OBS_station import start_time, end_time, client, LOCATION, NETWORK, CHANNEL, station_data

STATION, lat, lon, elev = station_data
radius = 1.5  # Bán kính khoảng 220km
minmagni = 2.5 # Độ lớn của địa chấn

catalog = client.get_events(starttime=start_time,
                            endtime=end_time,
                            latitude=lat,
                            longitude=lon,
                            maxradius=radius,
                            minmagnitude=minmagni)

# Sắp xếp theo thời gian tăng dần
catalog.events.sort(key=lambda e: e.origins[0].time)

if __name__ == "__main__":
    print(f"---🔍 Tổng cộng {len(catalog)} sự kiện---")

for i, event in enumerate(catalog):
    origin_time = event.origins[0].time
    magnitude = event.magnitudes[0].mag
    if __name__ == "__main__":
        print(f"{i:02d}. ⏰ {origin_time} | M = {magnitude}")
