# 🌳 Tree Species Classifier

A deep learning-based Streamlit web application to classify Indian tree species from images using a trained DenseNet121 model.

## 📌 Features

- Upload an image of a tree and get the predicted species.
- View the confidence score for each of the 29 classes.
- Interactive Streamlit interface with image preview and bar chart visualization.
- Basic login sidebar (name/email/password form).
- Ideal for botanists, students, environmental researchers, and nature enthusiasts.

## 🧠 Model Details

- **Model:** DenseNet121 (Transfer Learning)
- **Frameworks Used:** TensorFlow, Keras
- **Input Image Size:** 320x320
- **Number of Classes:** 29 tree species
- **Tree species:** Amla, Asopalav, Babul, Bamboo, Banyan, Bili, Cactus, Champa, Coconut, Garmalo, Gulmohor, Gunda, Jamun, Kanchan, Kesudo, Khajur, Mango, Motichanoti, Neem, Nilgiri, Pilikaren, Pipal, Saptaparni, Shirish, Simlo, Sitafal, Sonmahor, Sugarcane, Vad

## 🌐 Web App Interface

Built using [Streamlit](https://streamlit.io/), the interface allows:

- Image upload for prediction
- Live confidence chart using bar chart
- Sidebar for user login and project information

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tree-species-classifier.git
cd tree-species-classifier
