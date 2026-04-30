import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 🎨 Überraschungs-Design (dark + neon glow)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: #e2e8f0;
    }

    h1, h2, h3 {
        color: #38bdf8;
        text-align: left;
    }

    .herrscher {
        font-size: 22px;
        font-weight: bold;
        color: #facc15;
        margin-bottom: 20px;
    }

    /* Glow Effekt */
    .glow {
        color: #38bdf8;
        text-shadow: 0 0 10px #38bdf8, 0 0 20px #0ea5e9;
    }

    /* Fake Feuerwerk (sanfte Animation) */
    @keyframes glowPulse {
        0% { box-shadow: 0 0 5px #38bdf8; }
        50% { box-shadow: 0 0 25px #38bdf8; }
        100% { box-shadow: 0 0 5px #38bdf8; }
    }

    .firework {
        animation: glowPulse 2s infinite;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 👑 Begrüßung
st.title("YOLOv8 Analyse-System")
st.markdown('<div class="herrscher">Willkommen, Herrscher 👑</div>', unsafe_allow_html=True)

# 🎆 Feuerwerk Button
if st.button("🎆 Feuerwerk aktivieren, Herrscher"):
    st.markdown('<div class="firework">✨ Das System feiert Euch, Herrscher ✨</div>', unsafe_allow_html=True)
    st.balloons()

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Upload
uploaded_file = st.file_uploader("Bild hochladen, Herrscher", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Euer Bild", use_column_width=True)

    img_array = np.array(image)

    results = model(img_array)

    result_img = results[0].plot()
    st.image(result_img, caption="Analyse", use_column_width=True)

    st.subheader("Ergebnis, Herrscher:")

    # ❗ Immer eine Vermutung
    if len(results[0].boxes) == 0:
        st.write("👁️ Herrscher, meine Vermutung: Es könnte ein Objekt außerhalb meines Trainings sein, möglicherweise ein ungewöhnlicher Gegenstand.")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # KEIN unsicher mehr – nur Vermutung
            st.write(f"👁️ Herrscher, meine Analyse: Dies ist vermutlich **{label}** ({conf:.2f})")

else:
    st.info("Herrscher, ladet ein Bild hoch, damit ich analysieren kann 👁️")
