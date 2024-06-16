import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
import face_recognition
import numpy as np

feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

print(feature_list[0])  # Optional: Print the first feature for debugging

# upload_dir=os.environ.get('UPLOAD_DIR')
def save_uploaded_image(uploaded_image):
    try:
        # os.makedirs(upload_dir, exist_ok=True)
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    features = []
    if len(face_recognition.face_encodings(img)):
        features = face_recognition.face_encodings(img)[0]
    return features


def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(face_recognition.face_distance(feature_list, features))

    # Use np.argmin for more efficient index retrieval
    index_pos = np.argmin(similarity)
    return index_pos


st.title('Which Indian celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Save the image in a directory
    if save_uploaded_image(uploaded_image):
        # Load the image
        display_image = Image.open(uploaded_image)

        # Extract the features
        features = extract_features(os.path.join('uploads', uploaded_image.name))
        print(features)  # Optional: Print features for debugging

        # Recommend the celebrity
        if len(features):  # Check if features were extracted (face detected)
            index_pos = recommend(feature_list, features)
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('-'))
            predicted_actor = predicted_actor.split('.')[0]

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(display_image)
            with col2:
                st.header(f"Seems like {predicted_actor}")
                st.image(filenames[index_pos], width=300)
        else:  # No face detected, display error message
            st.error("There is an issue with the image. Please upload an image containing a face.")

else:
    st.info("Upload an image to see if it resembles an Indian celebrity!")