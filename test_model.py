import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, confusion_matrix

# ================================================
# ✅ 1) Load model CNN-LSTM đã huấn luyện
# ================================================
model = load_model('result_2/best_model_epoch09_valAcc0.9187.h5') 
print("✅ Model loaded!")

# ================================================
# ✅ 2) Chuẩn bị generator
# ================================================
test_gen = ImageDataGenerator(rescale=1./255)

# ✅ Thư mục cha chứa 2 folder con: earthquake/ & noise/
# Ví dụ:
# ├── test_model.py
# ├── earthquake/
# ├── noise/
test_dir = './'  # hoặc './test_data/' nếu bạn để riêng

img_size = (128, 128)
batch_size = 32

# ✅ CHỈ load 2 folder này!
test_data = test_gen.flow_from_directory(
    directory=test_dir,
    classes=['earthquake', 'noise'],  # ép keras chỉ quét đúng 2 folder
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

print(f"Class indices: {test_data.class_indices}")

# ================================================
# ✅ 3) Dự đoán
# ================================================
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_data.classes

# ================================================
# ✅ 4) Thống kê
# ================================================
eq_idx = list(test_data.class_indices.keys()).index('earthquake')
n_eq_pred = np.sum(y_pred == eq_idx)
n_noise_pred = len(y_pred) - n_eq_pred

n_eq_true = np.sum(y_true == eq_idx)
n_noise_true = len(y_true) - n_eq_true

print(f"🔍 Tổng số ảnh test: {len(y_true)}")
print(f"✅ Mô hình dự đoán Earthquake: {n_eq_pred}")
print(f"✅ Mô hình dự đoán Noise: {n_noise_pred}")
print(f"🎯 Thực tế: {n_eq_true} Earthquake | {n_noise_true} Noise")

# ================================================
# ✅ 5) Accuracy
# ================================================
acc = accuracy_score(y_true, y_pred)
print(f"🎯 Test Accuracy: {acc*100:.2f}%")

# ================================================
# ✅ 6) Confusion Matrix
# ================================================
cm = confusion_matrix(y_true, y_pred)
print("\n🧮 Confusion Matrix:")
print(cm)

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=test_data.class_indices.keys(),
    yticklabels=test_data.class_indices.keys()
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
