import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random

# 🎨 DESIGN (babyblau + lesbar + linksbündig)
st.markdown("""
<style>
.stApp {
    background: #e7f5ff;
    color: #0f172a;
}

/* Layout links */
.block-container {
    padding-top: 2rem;
    text-align: left;
}

/* Überschriften */
h1, h2, h3 {
    color: #1c7ed6;
    text-align: left;
}

/* Herrscher Text */
.herrscher {
    font-size: 22px;
    font-weight: bold;
    color: #0b7285;
}

/* Buttons */
.stButton > button {
    background-color: #74c0fc;
    color: #0b2239;
    border-radius: 10px;
    border: none;
}
.stButton > button:hover {
    background-color: #4dabf7;
}

/* kleine Kartenoptik */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# 👑 Begrüßung
st.title("YOLOv8 Objekterkennung")
st.markdown('<div class="herrscher">Willkommen, Herrscher 👑</div>', unsafe_allow_html=True)

# 🎆 FEUERWERK
def show_firework():
    colors = ["#ff0000", "#00ffcc", "#ffff00", "#ff00ff", "#00ff00"]
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
            animation: explode 1s ease-out forwards;">
        </div>
        """
    html += """
    <style>
    @keyframes explode {
        0% {transform: scale(1); opacity:1;}
        100% {transform: scale(12); opacity:0;}
    }
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)

# 🎈 BALLOON
def show_balloon(text):
    st.markdown(f"""
    <div style="
        position:fixed;
        bottom:-100px;
        left:50%;
        transform:translateX(-50%);
        font-size:28px;
        animation: floatUp 5s linear forwards;">
        🎈 {text}
    </div>

    <style>
    @keyframes floatUp {{
        0% {{bottom:-100px; opacity:0;}}
        50% {{opacity:1;}}
        100% {{bottom:100vh; opacity:0;}}
    }}
    </style>
    """, unsafe_allow_html=True)

# 🎮 BUTTONS
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

if st.button("🎆 Feuerwerk, Herrscher"):
    show_firework()

# 🤖 MODEL
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 🧠 FEEDBACK SPEICHER
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# 📤 UPLOAD
uploaded_file = st.file_uploader("Bild hochladen, Herrscher", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Euer Bild", use_column_width=True)

    img_array = np.array(image)
    results = model(img_array)

    result_img = results[0].plot()
    st.image(result_img, caption="Analyse", use_column_width=True)

    st.subheader("Ergebnis, Herrscher:")

    predictions = []

    if len(results[0].boxes) == 0:
        text = "👁️ Herrscher, ich vermute ein unbekanntes Objekt."
        st.write(text)
        predictions.append(text)

    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf >= 0.5:
                text = f"👑 Herrscher, das ist **{label}** ({conf:.2f})"
            else:
                text = f"🤔 Herrscher, ich vermute: **{label}** ({conf:.2f})"

            st.write(text)
            predictions.append(text)

    # 🧠 FEEDBACK
    st.markdown("### 🧠 Feedback, Herrscher")

    user_feedback = st.text_input("Was war wirklich richtig?")

    if st.button("Feedback speichern"):
        st.session_state.feedback.append({
            "prediction": predictions,
            "correct_answer": user_feedback
        })
        st.success("Feedback gespeichert, Herrscher 👑")

    # 📚 DATEN ANZEIGEN
    if st.session_state.feedback:
        st.markdown("### 📊 Gesammeltes Feedback")
        st.write(st.session_state.feedback)

else:
    st.info("Herrscher, bitte ladet ein Bild hoch 👁️")
