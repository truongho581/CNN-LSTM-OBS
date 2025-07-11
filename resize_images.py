from PIL import Image
import glob
import os

def resize_images(folder, size=(128, 128)):
    image_paths = glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)
    print(f"📁 Đang xử lý {len(image_paths)} ảnh trong '{folder}' (bao gồm subfolders)")

    for path in image_paths:
        try:
            img = Image.open(path).convert("L")
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(path)
        except Exception as e:
            print(f"⚠️ Lỗi xử lý {path}: {e}")

resize_images("earthquake")
resize_images("noise")
