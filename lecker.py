import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random

# 🎨 Kontrastreiches Design (besser lesbar + linksbündig)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #f1f5f9;
}

/* Alles links */
.block-container {
    padding-top: 2rem;
    text-align: left;
}

/* Überschriften */
h1, h2, h3 {
    color: #38bdf8;
    text-align: left;
}

/* Herrscher Style */
.herrscher {
    font-size: 22px;
    font-weight: bold;
    color: #fbbf24;
}

/* Boxen besser sichtbar */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# 👑 Begrüßung
st.title("YOLOv8 System")
st.markdown('<div class="herrscher">Willkommen, Herrscher 👑</div>', unsafe_allow_html=True)

# 🎆 Effekte
def show_firework():
    colors = ["#ff0000", "#00ffcc", "#ffff00", "#ff00ff", "#00ff00"]
    html = ""
    for _ in range(25):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        color = random.choice(colors)
        html += f'<div style="position:fixed; left:{x}vw; top:{y}vh; width:8px; height:8px; background:{color}; border-radius:50%; animation: explode 1s ease-out;"></div>'
    html += """
    <style>
    @keyframes explode {
        from {transform: scale(1); opacity:1;}
        to {transform: scale(12); opacity:0;}
    }
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)

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

if st.button("🎆 Feuerwerk, Herrscher"):
    show_firework()

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Speicher für Feedback (Session)
if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []

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

    prediction_texts = []

    if len(results[0].boxes) == 0:
        text = "👁️ Herrscher, meine Vermutung: Ein unbekanntes Objekt."
        st.write(text)
        prediction_texts.append(text)
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if conf >= 0.5:
                text = f"👑 Herrscher, das ist **{label}** ({conf:.2f})"
            else:
                text = f"🤔 Herrscher, ich vermute, es könnte **{label}** sein ({conf:.2f})"

            st.write(text)
            prediction_texts.append(text)

    # 🧠 Feedback Bereich
    st.markdown("### 🧠 Feedback geben, Herrscher")

    feedback = st.text_input("Was wäre die richtige Lösung gewesen?")

    if st.button("Feedback speichern"):
        st.session_state.feedback_list.append({
            "prediction": prediction_texts,
            "feedback": feedback
        })
        st.success("Feedback gespeichert, Herrscher 👑")

    # Anzeigen gespeicherter Daten
    if st.session_state.feedback_list:
        st.markdown("### 📚 Gesammeltes Wissen")
        st.write(st.session_state.feedback_list)

else:
    st.info("Herrscher, ladet ein Bild hoch 👁️")
