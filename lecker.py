import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random

# 🎨 Design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #1e293b, #0f172a);
    color: #e2e8f0;
}

/* Linksbündig */
h1, h2, h3 {
    text-align: left;
    color: #22d3ee;
}

.herrscher {
    font-size: 22px;
    font-weight: bold;
    color: #fbbf24;
}

/* Ballon Animation */
.balloon {
    position: fixed;
    bottom: -100px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 30px;
    animation: floatUp 5s linear forwards;
}

@keyframes floatUp {
    0% { bottom: -100px; opacity: 0; }
    50% { opacity: 1; }
    100% { bottom: 100vh; opacity: 0; }
}

/* Fake Feuerwerk */
.firework {
    position: fixed;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    animation: explode 1s ease-out forwards;
}

@keyframes explode {
    0% { transform: scale(1); opacity: 1; }
    100% { transform: scale(15); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# 👑 Begrüßung
st.title("YOLOv8 System")
st.markdown('<div class="herrscher">Willkommen, Herrscher 👑</div>', unsafe_allow_html=True)

# 🎆 Funktionen
def show_firework():
    colors = ["#ff0000", "#00ffcc", "#ffff00", "#ff00ff", "#00ff00"]
    html = ""
    for _ in range(20):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        color = random.choice(colors)
        html += f'<div class="firework" style="left:{x}vw; top:{y}vh; background:{color};"></div>'
    st.markdown(html, unsafe_allow_html=True)

def show_balloon(text):
    st.markdown(f'<div class="balloon">🎈 {text}</div>', unsafe_allow_html=True)

# 🎮 Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("drück mich herrscher"):
        show_balloon("I love Thore ❤️")

with col2:
    if st.button("i love informatik"):
        show_balloon("Informatik ist Liebe 💻❤️")

with col3:
    if st.button("überraschung"):
        if random.choice([True, False]):
            show_firework()
        else:
            show_balloon("👑 Du bist legendär, Herrscher")

# Extra Feuerwerk Button
if st.button("🎆 Feuerwerk, Herrscher"):
    show_firework()

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

    if len(results[0].boxes) == 0:
        st.write("👁️ Herrscher, meine Vermutung: Es könnte ein unbekanntes oder ungewöhnliches Objekt sein.")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf >= 0.5:
                st.write(f"👑 Herrscher, das ist **{label}** ({conf:.2f})")
            else:
                st.write(f"🤔 Herrscher, ich vermute, es könnte **{label}** sein ({conf:.2f})")

else:
    st.info("Herrscher, ladet ein Bild hoch 👁️")
