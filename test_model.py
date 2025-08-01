import os, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

IMG_SIZE = (64, 64)
MAX_FRAMES = 40

def parse_sequence(frames, label_file):
    label_text = tf.io.read_file(label_file)
    label_text = tf.strings.strip(label_text)
    label_chars = tf.strings.bytes_split(label_text)
    labels = tf.strings.to_number(label_chars, out_type=tf.float32)

    frames = frames[:MAX_FRAMES]
    labels = labels[:MAX_FRAMES]

    imgs = tf.map_fn(
        lambda f: tf.cast(
            tf.image.resize(tf.image.decode_png(tf.io.read_file(f), channels=1), IMG_SIZE),
            tf.float32) / 255.0,
        frames,
        fn_output_signature=tf.float32
    )

    seq_len = tf.shape(imgs)[0]
    pad_len = MAX_FRAMES - seq_len
    imgs = tf.pad(imgs, [[0, pad_len], [0, 0], [0, 0], [0, 0]])
    labels = tf.pad(tf.reshape(labels, [-1, 1]), [[0, pad_len], [0, 0]])

    imgs.set_shape([MAX_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 1])
    labels.set_shape([MAX_FRAMES, 1])

    return imgs, labels

# ==== Load model ====
model = load_model("best_model.h5")

# ==== Chọn sample ====
seq_path = "samples_test/M12B_sample_000"   # đổi thành folder của bạn
frames = sorted(glob.glob(os.path.join(seq_path, "*.png")))
label_file = os.path.join(seq_path, "label.txt")

# 🔑 Convert list -> Tensor trước khi parse
frames_tensor = tf.constant(frames, dtype=tf.string)
label_tensor = tf.constant(label_file, dtype=tf.string)

imgs, labels = parse_sequence(frames_tensor, label_tensor)

x = tf.expand_dims(imgs, axis=0)  # (1, T, 64, 64, 1)
y_true = labels[:, 0].numpy()

y_prob = model.predict(x, verbose=0)[0][:, 0]
y_pred = (y_prob > 0.5).astype(int)

print("✅ True labels:", y_true)
print("🔍 Predicted:", y_pred)
print(f"🎯 Frame-level accuracy: {np.mean(y_true == y_pred) * 100:.2f}%")

plt.figure(figsize=(12, 3))
plt.plot(y_true, 'g-o', label='True')
plt.plot(y_prob, 'b-x', label='Predicted Prob')
plt.axhline(0.5, color='r', linestyle='--', label='Threshold')
plt.ylim(-0.1, 1.1)
plt.xlabel("Frame index")
plt.ylabel("EQ Probability")
plt.legend()
plt.tight_layout()
plt.savefig("test_sample_result.png", dpi=300)
plt.show()
