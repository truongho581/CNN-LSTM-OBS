from seisbench.models.pickblue import PickBlue
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
import re

# 🧠 Load mô hình PickBlue với base EQTransformer
model = PickBlue(base="eqtransformer")

# 📥 Đọc waveform từ file .mseed
st = read("mseed_data/G30A/G30A_eq_002.mseed")
tr = st.select(channel="BHZ")[0]
starttime = tr.stats.starttime
fs = tr.stats.sampling_rate

channels = [tr.stats.channel for tr in st]
print("📦 Channels in file:", channels)

# Kiểm tra 3 thành phần Z, N, E có mặt không
has_3_components = all(any(ch.endswith(c) for ch in channels) for c in ["Z", "N", "E"])
print("✅ Đủ 3 thành phần Z-N-E:" if has_3_components else "❌ Thiếu thành phần")

# 🔍 Phân tích pick pha
result = model.classify(st)

# 🖨️ In pick (và parse thời gian từ string nếu cần)
print("📌 Picks:")
parsed_picks = []
for pick in result.picks:
    s = str(pick)
    
    # Tìm thời gian pick
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", s)
    if match:
        pick_time = UTCDateTime(match.group(0))
    else:
        print(f"⚠️ Không tìm được thời gian trong pick: {s}")
        continue

    if s.strip().endswith("P"):
        phase = "P"
    elif s.strip().endswith("S"):
        phase = "S"
    else:
        phase = "?"

    parsed_picks.append((pick_time, phase))
    print(f"{pick_time.isoformat()}     {phase}")


# 🎨 Vẽ spectrogram
f, t_spec, Sxx = spectrogram(tr.data, fs=fs, nperseg=256, noverlap=128)
Sxx_db = 10 * np.log10(Sxx + 1e-10)

plt.figure(figsize=(12, 6))
plt.pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap="viridis")
plt.ylim(2, 10)  # 🔍 Chỉ vẽ từ 2–10 Hz
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.colorbar(label="Power (dB)")


# 🟡 Vẽ đường origin time (start + 30s)
plt.axvline(x=30, color="white", linestyle="--", linewidth=2, label="Origin Time")

# 🟢 Vẽ pick P/S
drawn_labels = set()
for pick_time, phase in parsed_picks:
    t_offset = pick_time - starttime
    if t_offset < 0 or t_offset > tr.stats.npts / fs:
        continue
    color = "cyan" if phase == "P" else "lime"
    label = phase if phase not in drawn_labels else None
    plt.axvline(x=t_offset, color=color, linestyle="--", linewidth=2, label=label)
    drawn_labels.add(phase)

plt.legend()
plt.tight_layout()
plt.show()
