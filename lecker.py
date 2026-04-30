import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 🎨 Custom CSS (babyblau + Streifen)
st.markdown("""
    <style>
    .stApp {
        background: repeating-linear-gradient(
            45deg,
            #d0ebff,
            #d0ebff 20px,
            #e7f5ff 20px,
            #e7f5ff 40px
        );
    }
    h1, h2, h3 {
        color: #1c7ed6;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 👋 Begrüßung
st.title("YOLOv8 Objekterkennung")
st.markdown("## moin Goat! 🐐")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Bild hochladen
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Originalbild", use_column_width=True)

    img_array = np.array(image)

    # YOLO Vorhersage
    results = model(img_array)

    # Ergebnisbild
    result_img = results[0].plot()
    st.image(result_img, caption="Erkannte Objekte", use_column_width=True)

    st.subheader("Ergebnis:")

    # Wenn nichts erkannt wurde
    if len(results[0].boxes) == 0:
        st.warning("Ich konnte nichts eindeutig erkennen, aber ich bin mir unsicher 🤔")

    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf < 0.5:
                st.write(f"🤔 Ich bin mir unsicher... könnte **{label}** sein ({conf:.2f})")
            else:
                st.write(f"✅ Das ist sehr wahrscheinlich **{label}** ({conf:.2f})")

else:
    st.info("Bitte lade ein Bild hoch, damit ich etwas erkennen kann 👀")
