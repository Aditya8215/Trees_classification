import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd

st.set_page_config(page_title="Tree Classifier 🌳", layout="centered")

# Load the pre-trained model
try:
    model = load_model('model.h5')
except Exception as e:
    st.error("Error loading model: " + str(e))
    st.stop()

class_names = [
    'Amla', 'Asopalav', 'Babul', 'Bamboo', 'Banyan', 'Bili', 'Cactus', 'Champa', 'Coconut', 'Garmalo',
    'Gulmohor', 'Gunda', 'Jamun', 'Kanchan', 'Kesudo', 'Khajur', 'Mango', 'Motichanoti', 'Neem', 'Nilgiri',
    'Pilikaren', 'Pipal', 'Saptaparni', 'Shirish', 'Simlo', 'Sitafal', 'Sonmahor', 'Sugarcane', 'Vad'
]

def predict_img(image):
    image = image.resize((320, 320))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    Accuracy= predictions[predicted_index] * 100

    # Display result
    st.success(f"🌿 **Predicted Class:** {predicted_class}")
    st.info(f"📊 **Confidence:** {Accuracy:.2f}%")

    # Show all class probabilities as a bar chart
    prob_df = pd.DataFrame({'Tree Species': class_names, 'Confidence': predictions * 100})
    st.bar_chart(prob_df.set_index('Tree Species'))

    return predicted_class

# Header Section
st.markdown("<h1 style='text-align: center; color: green;'>🌳 Tree Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a tree image and let the AI identify the species.</p>", unsafe_allow_html=True)
st.image("https://cdn.shopify.com/s/files/1/0778/2679/files/banyan_c7f36ef7-751f-44b3-9891-364cb5a7f06f_1024x1024.jpg?16096657562656961956", width=600)

# Main Prediction Section
st.subheader("📷 Upload Tree Image")
uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image")
    if st.button("🔍 Predict"):
        predict_img(img)

# Sidebar: Login + Project Info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("🔐 Login")

with st.sidebar.form("login_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")
    if submitted:
        st.sidebar.success(f"Welcome, {name}!")
    

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌱 About the Project")
st.sidebar.write("""
This Tree Classifier project uses deep learning to identify different tree species from images. 
It helps students, researchers, and nature lovers to recognize plants and trees more easily.
""")

st.sidebar.markdown("""
- ✅ Trained on multiple Indian tree species
- 📦 Model: Transfer Learning and fined Tuned DenseNe121
- 🔍 Uses: Keras + TensorFlow
- 🌐 Web App via Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with  by Aditya Vashist")

# Hide sidebar collapse arrow
st.markdown("""
    <style>
    [data-testid="stSidebarNavCollapse"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)
