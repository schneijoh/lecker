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

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []

# ---------------------------
# ⏳ LOADING SCREEN
# ---------------------------
if not st.session_state.loaded:

    st.markdown("""
    <style>
    .stApp {
        background: #2b1d0e;
    }

    .loader {
        margin-top: 35vh;
        text-align: center;
        font-size: 40px;
        color: #f5e6c8;
        font-family: serif;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="loader">⏳ warte kurz goty...</div>', unsafe_allow_html=True)

    time.sleep(2)
    st.session_state.loaded = True
    st.rerun()

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
    }

    button {
        margin-top: 25px !important;
        background: #8b6b3f !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="book">
        <div class="title">FUNDGRUBE</div>
        <p>Altes Wissen erwacht...</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("drück mich goat"):
        st.session_state.opened = True
        st.rerun()

    st.stop()

# ---------------------------
# 🎆 FEUERWERK
# ---------------------------
firework_placeholder = st.empty()

def firework():
    colors = ["#ffcc00", "#ff0000", "#00ffff", "#ff00ff", "#ffffff"]
    html = ""
    for _ in range(25):
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
    firework_placeholder.markdown(html, unsafe_allow_html=True)

# ---------------------------
# 🎆 SPECIAL TEXTS
# ---------------------------
def love_text():
    st.markdown("""
    <div style="
        position:fixed;
        top:40vh;
        left:50%;
        transform:translateX(-50%);
        font-size:40px;
        font-family:serif;
        color:#5a3e1b;
        animation: fade 3s ease-out;">
        ❤️ ich liebe sie ❤️
    </div>

    <style>
    @keyframes fade {
        0% {opacity:0;}
        20% {opacity:1;}
        80% {opacity:1;}
        100% {opacity:0;}
    }
    </style>
    """, unsafe_allow_html=True)

def sound_text():
    st.markdown("""
    <div style="
        text-align:center;
        font-size:35px;
        color:#5a3e1b;
        font-family:serif;">
        🎵 johan ist der goat 🎵
    </div>
    """, unsafe_allow_html=True)
    time.sleep(5)

# ---------------------------
# 📖 FUNDGRUBE DESIGN
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: #e6d3b3;
    color: #2c1f14;
    font-family: serif;
}

.block-container {
    text-align: left;
}

h1, h2, h3 {
    color: #5a3e1b;
}

.herrscher {
    font-size: 22px;
    font-weight: bold;
}

.stButton > button {
    background-color: #8b6b3f;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("FUNDGRUBE")
st.markdown('<div class="herrscher">Willkommen, Herrscher 👑</div>', unsafe_allow_html=True)

# ---------------------------
# 🎮 BUTTONS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("klicken sie"):
        firework()
        love_text()

with col2:
    if st.button("sound"):
        sound_text()

# ---------------------------
# 🤖 MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------------------
# 📤 UPLOAD + ERKENNUNG
# ---------------------------
uploaded_file = st.file_uploader("Eintrag in die Fundgrube", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    results = model(np.array(image))

    st.image(results[0].plot(), use_column_width=True)

    st.subheader("Deutung, Herrscher:")

    predictions = []

    if len(results[0].boxes) == 0:
        text = "📜 Herrscher, unbekannter Fund."
        st.write(text)
        predictions.append(text)
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf >= 0.5:
                text = f"📖 eindeutig: **{label}** ({conf:.2f})"
            else:
                text = f"📜 vermutlich: **{label}** ({conf:.2f})"

            st.write(text)
            predictions.append(text)

    # ---------------------------
    # 🧠 FEEDBACK SYSTEM
    # ---------------------------
    st.markdown("### 🧠 Feedback, Herrscher")

    feedback = st.text_input("Was war wirklich richtig?")

    if st.button("Feedback speichern"):
        st.session_state.feedback_list.append({
            "prediction": predictions,
            "feedback": feedback
        })
        st.success("Feedback gespeichert, Herrscher 👑")

    if st.session_state.feedback_list:
        st.markdown("### 📚 Archiv der Fundgrube")
        st.write(st.session_state.feedback_list)

else:
    st.info("Herrscher, bitte einen Fund einreichen 📖")
