import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random
import time

# ---------------------------
# STATE INIT
# ---------------------------
if "opened" not in st.session_state:
    st.session_state.opened = False

# ---------------------------
# BOOK INTRO SCREEN
# ---------------------------
if not st.session_state.opened:

    st.markdown("""
    <style>
    .stApp {
        background: #2b1d0e;
    }

    .book {
        width: 90vw;
        height: 85vh;
        margin: auto;
        margin-top: 5vh;
        background: linear-gradient(90deg, #5a3e1b, #3b2912);
        border-radius: 20px;
        box-shadow: 0 0 40px rgba(0,0,0,0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        color: #f5e6c8;
        font-family: serif;
        text-align: center;
    }

    .title {
        font-size: 40px;
        margin-bottom: 20px;
    }

    button {
        background: #c9a66b !important;
        color: black !important;
        font-size: 18px !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="book">
        <div class="title">📖 Das Buch des Herrschers 📖</div>
        <p>Ein altes Wissen liegt verborgen...</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("drück mich goat"):
        st.session_state.opened = True
        st.rerun()

    st.stop()

# ---------------------------
# FEUERWERK INTRO (10s)
# ---------------------------
placeholder = st.empty()

def firework_frame():
    colors = ["#ff0000", "#00ffcc", "#ffff00", "#ff00ff", "#00ff00"]
    html = ""
    for _ in range(40):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        color = random.choice(colors)
        html += f"""
        <div style="
            position:fixed;
            left:{x}vw;
            top:{y}vh;
            width:10px;
            height:10px;
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

# 10 Sekunden Feuerwerk nur einmal
if "firework_done" not in st.session_state:
    st.session_state.firework_done = True

    for _ in range(10):
        firework_frame()
        time.sleep(1)

    placeholder.empty()

# ---------------------------
# APP DESIGN
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: #e7f5ff;
    color: #0f172a;
}
.block-container {
    text-align: left;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #1c7ed6;
}
.herrscher {
    font-size: 22px;
    font-weight: bold;
    color: #0b7285;
}
</style>
""", unsafe_allow_html=True)

st.title("YOLOv8 System")
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
uploaded_file = st.file_uploader("Bild hochladen, Herrscher", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    img_array = np.array(image)
    results = model(img_array)

    st.image(results[0].plot(), use_column_width=True)

    st.subheader("Ergebnis, Herrscher:")

    if len(results[0].boxes) == 0:
        st.write("👁️ Herrscher, ich vermute etwas Unbekanntes im Bild.")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf >= 0.5:
                st.write(f"👑 Herrscher: **{label}** ({conf:.2f})")
            else:
                st.write(f"🤔 Herrscher, ich vermute: **{label}** ({conf:.2f})")

else:
    st.info("Herrscher, bitte ladet ein Bild hoch 👁️")
    
