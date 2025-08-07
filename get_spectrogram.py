import os
import re
import numpy as np
from obspy import UTCDateTime, read
from scipy.signal import spectrogram
from seisbench.models.pickblue import PickBlue
import matplotlib.pyplot as plt

# ==== Cấu hình ====
BASE_DIR = "mseed_data"
SAVE_DIR = "spectrogram_data"
os.makedirs(SAVE_DIR, exist_ok=True)

model = PickBlue(base="eqtransformer")

# ==== Duyệt qua tất cả trạm có thư mục trong mseed_data/ ====
stations = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

for STATION in sorted(stations):
    station_dir = os.path.join(BASE_DIR, STATION)
    mseed_files = sorted(f for f in os.listdir(station_dir) if f.endswith(".mseed") and "_eq_" in f)

    for fname in mseed_files:
        path = os.path.join(station_dir, fname)
        try:
            print(f"\n📡 Trạm {STATION} | File: {fname}")
            st = read(path)
            st.detrend("demean")
            st.filter("bandpass", freqmin=2, freqmax=10)

            # Kiểm tra đủ 3 kênh BHZ/BHN/BHE
            if not all(st.select(channel=f"BH{c}") for c in ["Z", "N", "E"]):
                print("⚠️ Thiếu BHZ/BHN/BHE → bỏ qua")
                continue

            stream = st.select(channel="BH*")
            trZ = stream.select(channel="BHZ")[0]
            fs = trZ.stats.sampling_rate
            trace_start = trZ.stats.starttime

            # ==== Pick pha bằng PickBlue ====
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
                print("⚠️ Không pick được pha nào → bỏ qua")
                continue

            # ==== Spectrogram BHZ ====
            f, t_spec, Sxx = spectrogram(trZ.data, fs=fs, nperseg=256, noverlap=128)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            # ==== Vẽ hình với overlay ====
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(t_spec, f, Sxx_db, shading="gouraud", cmap="viridis", vmin=-80, vmax=80)
            plt.ylabel("Frequency (Hz)")
            plt.ylim(2, 10)
            plt.xlabel("Time (s)")
            plt.colorbar(label="Power (dB)")
            plt.axvline(x=30, color="white", linestyle="--", linewidth=2, label="Start Time")

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

            # ==== Lưu ảnh giữ nguyên tên ====
            out_name = os.path.splitext(fname)[0] + ".png"  # giữ tên như M12B_eq_000.png
            out_path = os.path.join(SAVE_DIR, STATION, out_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=300)
            plt.close()

            print(f"✅ Đã lưu {out_name}")

        except Exception as e:
            print(f"❌ Lỗi xử lý {fname}: {e}")
            continue
