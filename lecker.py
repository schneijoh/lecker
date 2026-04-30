import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# 🎨 Babyblaues Design
st.markdown("""
    <style>
    .stApp {
        background-color: #e7f5ff;
    }
    h1, h2, h3 {
        color: #1c7ed6;
        text-align: center;
    }
    .herrscher {
        font-size: 20px;
        font-weight: bold;
        color: #1864ab;
    }
    </style>
""", unsafe_allow_html=True)

# 👑 Begrüßung
st.title("YOLOv8 Objekterkennung")
st.markdown('<p class="herrscher">Moin Herrscher! 👑</p>', unsafe_allow_html=True)

# 🎆 Feuerwerk Button
if st.button("🎆 Feuerwerk starten, Herrscher!"):
    st.balloons()  # eingebauter Effekt
    st.success("Ein Feuerwerk zu Ehren des Herrschers! 🎇")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload
uploaded_file = st.file_uploader("Lade ein Bild hoch, Herrscher", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Euer Bild, Herrscher", use_column_width=True)

    img_array = np.array(image)

    results = model(img_array)

    result_img = results[0].plot()
    st.image(result_img, caption="Analyse für den Herrscher", use_column_width=True)

    st.subheader("Ergebnis, Herrscher:")

    # ❗ Immer etwas sagen
    if len(results[0].boxes) == 0:
        st.warning("Herrscher, ich bin unsicher… aber es könnte ein Objekt sein, das ich nicht klar erkennen kann 🤔")

    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf < 0.5:
                st.write(f"🤔 Herrscher, ich bin mir nicht sicher… aber ich tippe auf **{label}** ({conf:.2f})")
            else:
                st.write(f"👑 Herrscher, hier wurde sehr wahrscheinlich **{label}** erkannt ({conf:.2f})")

else:
    st.info("Herrscher, bitte ladet ein Bild hoch 👀")import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# 🎨 Babyblaues Design
st.markdown("""
    <style>
    .stApp {
        background-color: #e7f5ff;
    }
    h1, h2, h3 {
        color: #1c7ed6;
        text-align: center;
    }
    .herrscher {
        font-size: 20px;
        font-weight: bold;
        color: #1864ab;
    }
    </style>
""", unsafe_allow_html=True)

# 👑 Begrüßung
st.title("YOLOv8 Objekterkennung")
st.markdown('<p class="herrscher">Moin Herrscher! 👑</p>', unsafe_allow_html=True)

# 🎆 Feuerwerk Button
if st.button("🎆 Feuerwerk starten, Herrscher!"):
    st.balloons()  # eingebauter Effekt
    st.success("Ein Feuerwerk zu Ehren des Herrschers! 🎇")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload
uploaded_file = st.file_uploader("Lade ein Bild hoch, Herrscher", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Euer Bild, Herrscher", use_column_width=True)

    img_array = np.array(image)

    results = model(img_array)

    result_img = results[0].plot()
    st.image(result_img, caption="Analyse für den Herrscher", use_column_width=True)

    st.subheader("Ergebnis, Herrscher:")

    # ❗ Immer etwas sagen
    if len(results[0].boxes) == 0:
        st.warning("Herrscher, ich bin unsicher… aber es könnte ein Objekt sein, das ich nicht klar erkennen kann 🤔")

    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf < 0.5:
                st.write(f"🤔 Herrscher, ich bin mir nicht sicher… aber ich tippe auf **{label}** ({conf:.2f})")
            else:
                st.write(f"👑 Herrscher, hier wurde sehr wahrscheinlich **{label}** erkannt ({conf:.2f})")

else:
    st.info("Herrscher, bitte ladet ein Bild hoch 👀")
