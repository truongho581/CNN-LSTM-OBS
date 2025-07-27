import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, val_ratio=0.2, seed=42):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Lấy list thư mục sample_xxx
    all_samples = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    random.seed(seed)
    random.shuffle(all_samples)

    val_size = int(len(all_samples) * val_ratio)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    # Copy thư mục sample
    for s in train_samples:
        src = os.path.join(source_dir, s)
        dst = os.path.join(train_dir, s)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    for s in val_samples:
        src = os.path.join(source_dir, s)
        dst = os.path.join(val_dir, s)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"✅ {len(train_samples)} samples train, {len(val_samples)} samples val cho {os.path.basename(source_dir)}")

# Đường dẫn thư mục ảnh gốc (chưa chia)
split_data(f"earthquake", "data/train/earthquake", "data/val/earthquake")
split_data(f"noise", "data/train/noise", "data/val/noise")


def create_test_set(source_dir, test_dir, num_test=50, seed=42):
    os.makedirs(test_dir, exist_ok=True)

    # Lấy list folder sample
    all_samples = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    random.seed(seed)
    random.shuffle(all_samples)

    test_samples = all_samples[:num_test]

    for s in test_samples:
        src = os.path.join(source_dir, s)
        dst = os.path.join(test_dir, s)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    print(f"✅ Đã tạo {len(test_samples)} samples test cho {os.path.basename(source_dir)}")

# Dùng cho 2 lớp
# create_test_set("earthquake", "data/test/earthquake", num_test=158)
# create_test_set("noise", "data/test/noise", num_test=158)

