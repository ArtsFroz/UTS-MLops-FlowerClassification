import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "resnet50_flower_full.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

IMG_WIDTH = 136
IMG_HEIGHT = 102
IMG_SIZE_PIL = (IMG_WIDTH, IMG_HEIGHT)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + convert ke array + preprocess_input untuk ResNet50."""
    # pastikan 3 channel
    image = image.convert("RGB")
    # resize ke ukuran training
    image = image.resize(IMG_SIZE_PIL)
    # ke numpy, float32
    img = np.array(image, dtype="float32")
    # tambah dimensi batch -> (1, H, W, 3)
    img = np.expand_dims(img, axis=0)
    # ResNet50 expect 0–255 lalu preprocess_input
    # gambar PIL sudah 0–255 -> langsung preprocess_input
    img = preprocess_input(img)
    return img

def predict(image: Image.Image):
    if image is None:
        # kalau user belum upload apa-apa
        return {cls: 0.0 for cls in class_names}

    x = preprocess_image(image)
    preds = model.predict(x)[0]  # shape (5,)
    idx = int(np.argmax(preds))
    print("DEBUG preds:", preds)
    print("DEBUG argmax:", idx, "->", class_names[idx])

    # mapping ke dict {label: probabilitas}
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload gambar bunga"),
    outputs=gr.Label(num_top_classes=5, label="Prediksi"),
    title="Flower Classification",
    description="Model ResNet50 pretrained untuk klasifikasi 5 jenis bunga.",
)

if __name__ == "__main__":
    demo.launch()
