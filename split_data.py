import os, shutil, random

SOURCE_DIR = "data"
OUTPUT_DIR = "data_split"
TRAIN_RATIO = 0.64
VAL_RATIO = 0.16
TEST_RATIO = 0.2

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

eq_samples = []
noise_samples = []

for station in os.listdir(SOURCE_DIR):
    st_path = os.path.join(SOURCE_DIR, station)
    if not os.path.isdir(st_path):
        continue
    for sample in os.listdir(st_path):
        sample_path = os.path.join(st_path, sample)
        if os.path.isdir(sample_path):
            if "eq_" in sample:
                eq_samples.append((station, sample_path))
            elif "noise_" in sample:
                noise_samples.append((station, sample_path))

print(f"🌋 EQ: {len(eq_samples)} | 🎵 Noise: {len(noise_samples)}")

# ==== Hàm split generic ====
def split_list(samples):
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return samples[:n_train], samples[n_train:n_train+n_val], samples[n_train+n_val:]

eq_train, eq_val, eq_test = split_list(eq_samples)
noise_train, noise_val, noise_test = split_list(noise_samples)

train_samples = eq_train + noise_train
val_samples = eq_val + noise_val
test_samples = eq_test + noise_test

random.shuffle(train_samples)
random.shuffle(val_samples)
random.shuffle(test_samples)

print(f"✅ Train: {len(train_samples)} (EQ={len(eq_train)}, Noise={len(noise_train)})")
print(f"✅ Val: {len(val_samples)} (EQ={len(eq_val)}, Noise={len(noise_val)})")
print(f"✅ Test: {len(test_samples)} (EQ={len(eq_test)}, Noise={len(noise_test)})")

def copy_samples(sample_list, split_name):
    for station, src_path in sample_list:
        sample_name = os.path.basename(src_path)
        new_name = f"{station}_{sample_name}"
        dst_dir = os.path.join(OUTPUT_DIR, split_name, new_name)
        shutil.copytree(src_path, dst_dir)

copy_samples(train_samples, "train")
copy_samples(val_samples, "val")
copy_samples(test_samples, "test")

print("✅ Đã chia dữ liệu và cân bằng EQ/Noise trong mỗi tập.")
