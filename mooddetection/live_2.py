import cv2
from keras.models import model_from_json
import numpy as np
import pandas as pd
import streamlit as st

# Load the image
image_path = "https://www.languagelearninginstitute.com/wp-content/uploads/2019/06/Music-Language.jpg"

# Display the image along with other content
st.markdown(
    f"""
    <div style='background-image: url("{image_path}"); height: 500px; background-size: cover; padding: 20px;'>
   
    </div>
    """,
    unsafe_allow_html=True
)

# Load the emotion detection model
json_file = open("emotion_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotion_model.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features for emotion prediction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to recommend songs based on detected emotion
def recommend_songs(emotion):
    csv_file_path = f"{emotion}.csv"
    try:
        df = pd.read_csv(csv_file_path)
        recommended_songs = df[['Name', 'Album', 'Artist']].values.tolist()
        return recommended_songs
    except FileNotFoundError:
        return None

# Streamlit app
st.title("Emotion-based Music Recommender")

# Function to process video frames
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try: 
        for (p, q, r, s) in faces:
            face_image = gray[q:q+s, p:p+r]
            cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
            face_image = cv2.resize(face_image, (48, 48))
            features = extract_features(face_image)
            pred = model.predict(features)
            emotion_label = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}[pred.argmax()]
            
            # Display the detected emotion
            cv2.putText(frame, '%s' % emotion_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            
            return emotion_label
    
    except cv2.error:
        pass

    return None

# Button to start emotion recognition
start_button = st.button("Start Emotion Recognition")

if start_button:
    # Open the camera and process the frames
    camera = cv2.VideoCapture(0)

    if camera.isOpened():
        while True:
            ret, frame = camera.read()
            if ret:
                captured_emotion = process_frame(frame)
                st.image(frame, channels="BGR")

                if captured_emotion:
                    st.write(f"Detected Emotion: {captured_emotion}")
                    recommended_songs = recommend_songs(captured_emotion)
                    if recommended_songs is not None:
                        st.write(f"Recommended songs for {captured_emotion}:")
                        recommended_songs_df = pd.DataFrame(recommended_songs, columns=['Name', 'Album', 'Artist'])
                        st.write(recommended_songs_df)
                    else:
                        st.warning("No songs found for this emotion.")
                    break  # Stop the loop after detecting and displaying emotion

        camera.release()
    else:
        st.warning("Unable to open camera")
