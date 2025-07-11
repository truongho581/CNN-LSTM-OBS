
import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client("IRIS")

# Các thông tin cho trạm OBS
time = ['2011-10-20', '2012-07-18']
start_time = UTCDateTime(time[0])
end_time = UTCDateTime(time[1])
NETWORK = "7D"
CHANNEL = "BHZ"
LOCATION = "--"

#station_data =  [station name, lat, lon, elevation]

# # Trạm để lấy data test
##station_data = ['M12B', 42.1840, -124.9461, -1045.0]  # '2012-09-02', '2013-06-18'
##station_data = ['M14B', 40.9850, -124.5897, -638.0]  # '2012-09-02', '2013-06-19'
##station_data = ['M16D', 41.6618, -124.8071, -882.0]  # '2014-08-12', '2015-09-17'

##station_data = ['M02C', 48.3069, -125.6012, -141.0]  # '2013-08-19', '2014-05-30'
station_data = ['M08A', 44.1187, -124.8953, -126.4]  # '2011-10-20', '2012-07-18'
# station_data = ['M08C', 44.1185, -124.8954, -131.0]  # '2013-08-21', '2014-06-01'

# # Trạm để lấy data train/val
##station_data = ['FS01B', 40.3268, -124.9492, -940.0]  # '2012-09-02', '2013-06-19'
##station_data = ['FS02D', 40.3260, -124.8002, -947.9]  # '2014-07-16', '2015-08-28'
##station_data = ['FS10D', 40.4328, -124.6940, -1153.8]  # '2014-07-17', '2015-08-27'
##station_data = ['FS16D', 40.5378, -124.7468, -1080.4]  # '2014-07-16', '2015-08-28'
##station_data = ['FS41D', 40.6124, -124.7310, -1079.3]  # '2014-07-16', '2015-08-28'
##station_data = ['FS44D', 40.7609, -124.7028, -837.0]  # '2014-08-12', '2015-09-15'
##station_data = ['G01D', 39.9999, -124.6008, -1006.7]  # '2014-07-17', '2015-08-27'
##station_data = ['M11B', 42.9320, -125.0171, -1109.0]  # '2012-09-02', '2013-06-18'
##station_data = ['G09D', 40.6665, -124.7269, -716.3]  # '2014-07-16', '2015-08-28'
##station_data = ['FS04D', 40.2528, -124.5052, -155.0]  # '2014-08-12', '2015-09-15'
##station_data = ['FS08D', 40.3347, -124.4653, -132.0]  # '2014-08-12', '2015-09-15'
##station_data = ['FS14B', 40.4955, -124.5918, -107.0]  # '2012-09-02', '2013-06-19'
##station_data = ['J09B', 43.1510, -124.7270, -252.0]  # '2012-09-02', '2013-06-21'
##station_data = ['J25A', 44.4729, -124.6216, -142.8]  # '2011-10-21', '2012-07-18'
##station_data = ['J25C', 44.4730, -124.6217, -144.0]  # '2013-08-22', '2014-06-01'
##station_data = ['J33C', 45.1068, -124.5708, -354.0]  # '2013-08-22', '2014-05-29'
##station_data = ['J65C', 47.8913, -125.1398, -169.0]  # '2013-08-19', '2014-05-29'
##station_data = ['J73A', 48.7677, -126.1925, -143.3]  # '2011-10-18', '2012-07-17'
##station_data = ['J73C', 48.7679, -126.1926, -133.0]  # '2013-08-19', '2014-05-30'
##station_data = ['M01A', 49.1504, -126.7221, -132.9]  # '2011-10-18', '2012-07-16'
##station_data = ['M01C', 49.1504, -126.7222, -138.0]  # '2013-08-19', '2014-05-30'
##station_data = ['M02A', 48.3070, -125.6004, -139.0]  # '2011-10-18', '2012-07-16'

## Test model
# station_data = ['G33D', 42.6653, -124.8020, -686.0]  # '2014-08-12', '2015-09-17'
# station_data = ['M04A', 47.5581, -125.1922, -563.4]  # '2011-10-17', '2012-07-16'
# station_data = ['M04C', 47.5584, -125.1923, -570.0]  # '2013-08-19', '2014-05-29'
# station_data = ['M05A', 46.1735, -124.9346, -828.2]  # '2011-10-17', '2012-07-15'
# station_data = ['M05C', 46.1735, -124.9345, -837.0]  # '2013-08-19', '2014-05-29'

# Hàm lấy danh sách trạm OBS
def get_station_info():
    inv = client.get_stations(
        network=NETWORK,
        station="*",
        channel=CHANNEL,
        starttime=start_time,
        endtime=end_time,
        level="channel"
    )

    stations = []
    network_info = ""

    for net in inv:
        network_info = f"{net.code} - {net.description}"
        for sta in net:
            lat = f"{sta.latitude:.4f}"
            lon = f"{sta.longitude:.4f}"
            elev = sta.elevation

            start_dates = []
            end_dates = []

            for chan in sta:
                if chan.start_date:
                    start_dates.append(chan.start_date)
                if chan.end_date:
                    end_dates.append(chan.end_date)

            sta_start = min(start_dates).strftime("%Y-%m-%d") if start_dates else None
            sta_end = max(end_dates).strftime("%Y-%m-%d") if end_dates else None

            if elev is not None and elev > -500:
                stations.append(
                    (sta.code, lat, lon, elev, sta_start, sta_end)
                )

    return network_info, stations



if __name__ == "__main__":
    print(f"📡 Trạm OBS thuộc mạng {NETWORK} có kênh {CHANNEL} trong khu vực:")

    network_info, station_list = get_station_info()
    print("📘 Network Info:", network_info)

    for sta_code, lat, lon, elev, sta_start, sta_end in station_list:
        print(f"station_data = ['{sta_code}', {lat}, {lon}, {elev}]  # '{sta_start}', '{sta_end}'")

