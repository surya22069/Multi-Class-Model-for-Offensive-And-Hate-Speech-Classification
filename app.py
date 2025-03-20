import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model("multi_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define max sequence length (should match training settings)
max_length = 50  # Ensure this matches the training sequence length

# Labels for each output category
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
violence_labels = ['sexual_violence', 'physical_violence', 'emotional_violence', 'Harmful_traditional_practice', 'economic_violence']
hate_labels = ['offensive speech', 'Neither', 'Hate Speech']

# 🎨 Streamlit UI
st.set_page_config(page_title="Text Classification", page_icon="🤖", layout="wide")
st.title("📌 Multi-Label Text Classification")
st.markdown("#### 🚀 Predict **Emotion**, **Violence**, and **Hate Speech** from a single input!")

# User input
user_input = st.text_area("📝 Enter Text:", placeholder="Type or paste your text here...", height=150)

if st.button("🔍 Classify Text"):
    if user_input:
        # Preprocess input
        input_sequence = tokenizer.texts_to_sequences([user_input])
        input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')

        # Pass the same input to all three branches
        predictions = model.predict({
            'emotion_input': input_padded,
            'violence_input': input_padded,
            'hate_input': input_padded
        })

        # Get predicted probabilities for each category
        emotion_prob = np.max(predictions[0])  # Highest probability for emotion
        violence_prob = np.max(predictions[1])  # Highest probability for violence
        hate_prob = np.max(predictions[2])  # Highest probability for hate speech

        # Determine the Major Label based on the highest probability
        probabilities = [emotion_prob, violence_prob, hate_prob]
        major_labels = ['Emotion', 'Violence', 'Hate']
        major_label_index = np.argmax(probabilities)
        major_label = major_labels[major_label_index]

        # Get the sub-label from the chosen category
        if major_label == 'Emotion':
            sub_label = emotion_labels[np.argmax(predictions[0])]
        elif major_label == 'Violence':
            sub_label = violence_labels[np.argmax(predictions[1])]
        else:
            sub_label = hate_labels[np.argmax(predictions[2])]

        # 🎯 Display Results
        st.success(f"🎯 **Category:** {major_label}")
        st.info(f"📌 **Sub Label:** {sub_label}")
    else:
        st.warning("⚠️ Please enter text before classifying!")

# Footer
st.markdown("---")
st.markdown("🚀 Developed with ❤️ using **Streamlit & TensorFlow**")
