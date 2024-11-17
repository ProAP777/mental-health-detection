import streamlit as st
import librosa
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Step 1: Recreate and Save Label Encoder
def recreate_label_encoder():
    labels = ['Normal', 'Anxious', 'Depressed']
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    joblib.dump(label_encoder, 'label_encoder.pkl')
    return label_encoder

# Function to Extract Features
def extract_features(file, segment, transcript, user_age):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)
    
    # Feature extraction
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    loudness = np.mean(librosa.feature.rms(y=y))
    
    # Pitch estimation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean([pitches[:, i].max() for i in range(pitches.shape[1]) if magnitudes[:, i].max() > 0])
    
    hnr = np.mean(librosa.effects.harmonic(y) / librosa.effects.percussive(y))
    
    # Placeholder for formant features (needs specialized tools)
    formant_f1, formant_f2, formant_f3 = 0, 0, 0
    
    # Speech duration
    speech_duration = librosa.get_duration(y=y, sr=sr)
    
    # Pauses (based on low energy segments)
    rms = librosa.feature.rms(y=y)
    pauses = len(np.where(rms[0] < 0.01)[0])  # Threshold for pauses
    
    # Placeholder features (replace with actual computations or models)
    sentiment = 0
    complexity = 0
    word_frequency = 0
    keywords = 0
    emotions = 0
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_features = [np.mean(mfccs[i]) for i in range(13)]
    
    # Encode `segment` and `transcript`
    segment_encoded = hash(segment) % 1000  # Dummy encoding
    transcript_encoded = hash(transcript) % 1000  # Dummy encoding
    
    # Combine all features into a single array, including user_age
    feature_vector = [
        segment_encoded,  # Encoded segment
        zcr_mean, spectral_centroid_mean, spectral_rolloff_mean, spectral_flux,
        loudness, pitch, hnr, formant_f1, formant_f2, formant_f3,
        speech_duration, pauses, sentiment, complexity, word_frequency, keywords,
        emotions, transcript_encoded,  # Encoded transcript
        user_age  # New feature
    ] + mfcc_features
    
    return feature_vector

# Load trained model
model = joblib.load('best_model.pkl')  # Replace with your actual model file

# Recreate or load label encoder
try:
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.warning("Label encoder not found, recreating it with default labels.")
    label_encoder = recreate_label_encoder()

# Streamlit App
st.title("🎤 Speech-Based Mental Health Detection System")

# Store user information and audio in session state to prevent reset
if 'user_info_submitted' not in st.session_state:
    st.session_state.user_info_submitted = False

if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None

# User Input: Name, Age, and Email
if not st.session_state.user_info_submitted:
    with st.form("user_input_form"):
        st.subheader("Enter Your Information")
        name = st.text_input("Enter Your Name")
        age = st.number_input("Enter Your Age", min_value=0, max_value=100, value=25, step=1)
        email = st.text_input("Enter Your Email")
        
        # Submit button
        submit_button = st.form_submit_button(label="Submit Information")

    if submit_button:
        st.session_state.user_name = name
        st.session_state.user_age = age
        st.session_state.user_email = email
        st.session_state.user_info_submitted = True
        st.write(f"Hello, {name}! Age: {age}, Email: {email}")

# Upload audio file after user information submission
if st.session_state.user_info_submitted:
    audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])
    st.session_state.audio_file = audio_file  # Store the uploaded file in session state

    if audio_file is not None:
        try:
            # Set segment and transcript values to default since we are not collecting them from the user
            segment = "Unknown"
            transcript = "No transcript provided"
            
            # Extract features
            features = extract_features(audio_file, segment, transcript, st.session_state.user_age)
            
            # Ensure the feature vector matches the expected size
            if len(features) != 33:
                st.error(f"Feature vector size mismatch! Extracted {len(features)} features, but model requires 33.")
                st.stop()
            
            # Convert to NumPy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Predict using the trained model
            prediction = model.predict(features_array)
            prediction_label = label_encoder.inverse_transform(prediction)
            
            # Display prediction
            st.success(f"Prediction: {prediction_label[0]}")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# Debugging: Show labels in the label encoder
if st.checkbox("Show Label Encoder Info"):
    st.write("Labels in the Label Encoder:")
    st.write(label_encoder.classes_)
