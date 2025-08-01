import os
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.signal import spectrogram
from PIL import Image
import matplotlib.cm as cm

# ==== Config ====
client = Client("IRIS")
NETWORK = "7D"
STATION = "M12B"
LOCATION = "--"

start_time = UTCDateTime("2012-09-02")
end_time = UTCDateTime("2013-06-18")

VMIN = -80
VMAX = 60
TARGET_SIZE = (512, 256)

os.makedirs("spectrogram_out", exist_ok=True)

# ==== Catalog sự kiện ====
catalog = client.get_events(starttime=start_time, endtime=end_time,
                            latitude=42.1840, longitude=-124.9461,
                            maxradius=1.5, minmagnitude=2.5)
catalog.events.sort(key=lambda e: e.origins[0].time)
print(f"📌 Tìm thấy {len(catalog)} sự kiện cho {STATION}")

# ==== Hàm tạo spectrogram dạng mảng [0–255] ====
def make_spec(tr, fs):
    nperseg = int(5.12 * fs)
    noverlap = int(0.5 * nperseg)
    f, t, Sxx = spectrogram(tr.data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power_db = 10 * np.log10(Sxx + 1e-10)
    f_mask = (f >= 2) & (f <= 10)
    power_db = power_db[f_mask, :]
    power_db = np.clip((power_db - VMIN) / (VMAX - VMIN), 0, 1)
    img = (power_db * 255).astype(np.uint8)
    img = img[::-1, :]  # ✅ Lật để tần số thấp ở dưới
    return np.array(Image.fromarray(img).resize(TARGET_SIZE))


# ==== Hàm lưu ảnh màu heatmap ====
def save_colormap_image(spec_array, out_path, cmap_name="inferno"):
    cmap = cm.get_cmap(cmap_name)
    rgba_img = (cmap(spec_array / 255.0)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgba_img).save(out_path)

# ==== Loop qua event ====
for i, event in enumerate(catalog):
    origin_time = event.origins[0].time
    print(f"\n🔍 Event {i}: {origin_time}")

    try:
        st = client.get_waveforms(NETWORK, STATION, LOCATION, "BH?",
                                  starttime=origin_time - 60,
                                  endtime=origin_time + 420)
        st.detrend("demean")
        st.filter("bandpass", freqmin=1, freqmax=10)

        channels = {tr.stats.channel: tr for tr in st}
        if all(x in channels for x in ["BHZ","BHN","BHE"]):
            trZ, trH1, trH2 = channels["BHZ"], channels["BHN"], channels["BHE"]
            label = "ZNE"
        elif all(x in channels for x in ["BHZ","BH1","BH2"]):
            trZ, trH1, trH2 = channels["BHZ"], channels["BH1"], channels["BH2"]
            label = "Z12"
        else:
            print(f"⚠️ {STATION} thiếu kênh ngang, bỏ event {i}")
            continue

        fs = trZ.stats.sampling_rate

        specZ = make_spec(trZ, fs)
        specH1 = make_spec(trH1, fs)
        specH2 = make_spec(trH2, fs)

        # Lưu ảnh màu cho từng kênh
        save_colormap_image(specZ, f"spectrogram_out/{STATION}_eq_{i:03}_BHZ.png")
        save_colormap_image(specH1, f"spectrogram_out/{STATION}_eq_{i:03}_{trH1.stats.channel}.png")
        save_colormap_image(specH2, f"spectrogram_out/{STATION}_eq_{i:03}_{trH2.stats.channel}.png")

        # Lưu RGB 3 kênh (raw stack để dùng cho CNN)
        rgb = np.stack([specZ, specH1, specH2], axis=-1)
        rgb_path = f"spectrogram_out/{STATION}_eq_{i:03}_RGB_{label}.png"
        Image.fromarray(rgb).save(rgb_path)

        print(f"💾 Saved spectrograms (BHZ, {trH1.stats.channel}, {trH2.stats.channel}, RGB) for event {i}")

    except Exception as e:
        print(f"❌ Error event {i}: {e}")
        continue
