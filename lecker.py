import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Titel
st.title("YOLOv8 Objekterkennung mit Streamlit")

# Modell laden (vortrainiert)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # kleines, schnelles Modell
    return model

model = load_model()

# Bild hochladen
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Originalbild", use_column_width=True)

    # In numpy konvertieren
    img_array = np.array(image)

    # YOLO Vorhersage
    results = model(img_array)

    # Ergebnisbild mit Bounding Boxes
    result_img = results[0].plot()

    st.image(result_img, caption="Erkannte Objekte", use_column_width=True)

    # Optional: Labels anzeigen
    st.subheader("Erkannte Klassen:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        st.write(f"{label} ({conf:.2f})")
