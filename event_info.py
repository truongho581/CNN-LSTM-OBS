from obspy import read

# Thay bằng đường dẫn file mseed của bạn
file_path = "mseed_data/G03A/G03A_eq_006.mseed"

# Đọc file
st = read(file_path)

print(f"\n📄 File: {file_path}")
print(f"📚 Tổng số trace: {len(st)}\n")

for i, tr in enumerate(st):
    stats = tr.stats
    print(f"🔹 Trace {i+1}")
    print(f"  ➤ Network  : {stats.network}")
    print(f"  ➤ Station  : {stats.station}")
    print(f"  ➤ Location : {stats.location}")
    print(f"  ➤ Channel  : {stats.channel}")
    print(f"  ➤ Start    : {stats.starttime}")
    print(f"  ➤ End      : {stats.endtime}")
    print(f"  ➤ Sampling : {stats.sampling_rate} Hz")
    print(f"  ➤ Npts     : {stats.npts}")
    duration = stats.endtime - stats.starttime
    print(f"  ➤ Duration : {duration:.2f} s\n")
