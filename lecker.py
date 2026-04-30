import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random
import time

# ---------------------------
# STATE
# ---------------------------
if "opened" not in st.session_state:
    st.session_state.opened = False

# ---------------------------
# 📖 BOOK STARTSCREEN
# ---------------------------
if not st.session_state.opened:

    st.markdown("""
    <style>
    .stApp {
        background: #2b1d0e;
    }

    .book {
        width: 60vw;
        height: 70vh;
        margin: auto;
        margin-top: 10vh;
        background: linear-gradient(135deg, #d6c1a3, #b89b73);
        border: 4px solid #5a3e1b;
        border-radius: 15px;
        box-shadow: 0 0 40px rgba(0,0,0,0.7);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-family: serif;
        color: #3b2a1a;
        text-align: center;
    }

    .title {
        font-size: 50px;
        font-weight: bold;
        letter-spacing: 3px;
    }

    .subtitle {
        font-size: 18px;
        margin-top: 10px;
    }

    button {
        margin-top: 30px !important;
        background: #8b6b3f !important;
        color: #fff !important;
        border-radius: 8px !important;
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="book">
        <div class="title">FUNDGRUBE</div>
        <div class="subtitle">Altes Wissen, verborgen in der Tiefe der Zeit</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("drück mich goat"):
        st.session_state.opened = True
        st.rerun()

    st.stop()

# ---------------------------
# 🎆 FEUERWERK ÜBERGANG
# ---------------------------
placeholder = st.empty()

def firework():
    colors = ["#ffcc00", "#ff6600", "#ffffff", "#ff0000", "#00ffff"]
    html = ""
    for _ in range(30):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        color = random.choice(colors)
        html += f"""
        <div style="
            position:fixed;
            left:{x}vw;
            top:{y}vh;
            width:8px;
            height:8px;
            background:{color};
            border-radius:50%;
            animation: boom 1s ease-out;">
        </div>
        """
    html += """
    <style>
    @keyframes boom {
        0% {transform: scale(1); opacity:1;}
        100% {transform: scale(15); opacity:0;}
    }
    </style>
    """
    placeholder.markdown(html, unsafe_allow_html=True)

if "firework_done" not in st.session_state:
    st.session_state.firework_done = True
    for _ in range(8):
        firework()
        time.sleep(1)
    placeholder.empty()

# ---------------------------
# 📖 FUNDGRUBE DESIGN (MAIN APP)
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: #e6d3b3; /* altes Papier */
    color: #2c1f14;
    font-family: serif;
}

.block-container {
    text-align: left;
    padding-top: 2rem;
}

/* Titel */
h1, h2, h3 {
    color: #5a3e1b;
}

/* Herrscher Stil */
.herrscher {
    font-size: 22px;
    font-weight: bold;
    color: #3b2a1a;
}

/* Buttons altmodisch */
.stButton > button {
    background-color: #8b6b3f;
    color: white;
    border-radius: 6px;
    border: none;
}
.stButton > button:hover {
    background-color: #6f5330;
}
</style>
""", unsafe_allow_html=True)

st.title("FUNDGRUBE")
st.markdown('<div class="herrscher">Willkommen, Herrscher 👑</div>', unsafe_allow_html=True)

# ---------------------------
# MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------
# UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Eintrag in die Fundgrube laden", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    img_array = np.array(image)
    results = model(img_array)

    st.image(results[0].plot(), use_column_width=True)

    st.subheader("Deutung des Fundes, Herrscher:")

    if len(results[0].boxes) == 0:
        st.write("📜 Herrscher, ich deute: Ein unbekanntes Artefakt der Fundgrube.")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf >= 0.5:
                st.write(f"📖 Herrscher, eindeutig erkannt: **{label}** ({conf:.2f})")
            else:
                st.write(f"📜 Herrscher, Deutung: vermutlich **{label}** ({conf:.2f})")

else:
    st.info("Herrscher, fügt einen Eintrag zur Fundgrube hinzu 📖")
    
