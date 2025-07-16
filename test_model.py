from PIL import Image

img_path = 'F:/downloads/UET/CNN-LSTM_Project/data/train/noise/J73C_sample_006/frame_023.png'
img = Image.open(img_path)

print("Mode:", img.mode)     # "L" là grayscale, "RGB" là ảnh màu
print("Size:", img.size)