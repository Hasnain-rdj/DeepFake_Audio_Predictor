import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
from datasets import load_dataset
import soundfile as sf
import joblib

# --- Sidebar Navigation and Branding ---
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("AI Project Suite")
page = st.sidebar.radio("Navigation", ["Audio Deepfake Detection", "Software Defect Prediction"], help="Choose a prediction tool.")
st.sidebar.markdown("---")
st.sidebar.info("Developed by Mohd. Hasnain with ❤️ | Data Science 2K25")
st.sidebar.markdown("<small>Powered by Streamlit, scikit-learn, TensorFlow, and Hugging Face</small>", unsafe_allow_html=True)

# --- Load models and data ---
vectorizer = joblib.load("vectorizer.pkl")
logreg = joblib.load("logreg_model.pkl")
svm = joblib.load("svm_model.pkl")
perc = joblib.load("perc_model.pkl")
dnn = tf.keras.models.load_model("dnn_model.h5")
_df = pd.read_csv('dataset.csv')
label_cols = [col for col in _df.columns if col.startswith('type_')]
try:
    audio_vectorizer = joblib.load("audio_vectorizer.pkl")
    audio_logreg = joblib.load("audio_logreg_model.pkl")
    audio_svm = joblib.load("audio_svm_model.pkl")
    audio_perc = joblib.load("audio_perc_model.pkl")
    audio_dnn = tf.keras.models.load_model("audio_dnn_model.h5")
except Exception:
    audio_vectorizer = audio_logreg = audio_svm = audio_perc = audio_dnn = None

def predict_audio_deepfake(audio_bytes, model_choice_audio):
    import io
    import librosa
    import numpy as np
    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)
    features = audio_vectorizer.transform(mfcc_mean)
    if model_choice_audio == "SVM" and audio_svm is not None:
        pred = audio_svm.predict(features)[0]
        conf = float(audio_svm.predict_proba(features)[0][pred])
    elif model_choice_audio == "Logistic Regression" and audio_logreg is not None:
        pred = audio_logreg.predict(features)[0]
        conf = float(audio_logreg.predict_proba(features)[0][pred])
    elif model_choice_audio == "Perceptron" and audio_perc is not None:
        pred = audio_perc.predict(features)[0]
        conf = 1.0
    elif model_choice_audio == "DNN" and audio_dnn is not None:
        pred = int((audio_dnn.predict(features) > 0.5)[0][0])
        conf = float(audio_dnn.predict(features)[0][0])
    else:
        pred = "Unknown"
        conf = 0.0
    label = "Bonafide" if pred == 0 else "Deepfake"
    return label, conf

# --- Main UI Layout ---
st.markdown("""
<style>
/* Clean, minimal look */
.stApp { background: var(--background-color) !important; }
section[data-testid="stSidebar"] { background: var(--secondary-background-color) !important; }
.stButton>button { border-radius: 8px; font-weight: 600; }
.stDataFrame { border-radius: 8px; }
hr { border: 1px solid var(--primary-color); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

cube_html = '''
<style>
.cube-container {
    width: 200px;
    height: 200px;
    margin: 0 auto 30px auto;
    perspective: 600px;
}
.cube {
    width: 100px;
    height: 100px;
    position: relative;
    transform-style: preserve-3d;
    animation: rotateCube 5s infinite linear;
}
@keyframes rotateCube {
    from {transform: rotateX(0deg) rotateY(0deg);}
    to {transform: rotateX(360deg) rotateY(360deg);}
}
.cube-face {
    position: absolute;
    width: 100px;
    height: 100px;
    background: rgba(40, 60, 90, 0.7);
    border: 2px solid #3399ff;
    box-shadow: 0 0 20px #3399ff33;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2em;
    color: #fff;
    font-family: 'Segoe UI', sans-serif;
}
.cube-face.front  { transform: translateZ(50px); }
.cube-face.back   { transform: rotateY(180deg) translateZ(50px); }
.cube-face.right  { transform: rotateY(90deg) translateZ(50px); }
.cube-face.left   { transform: rotateY(-90deg) translateZ(50px); }
.cube-face.top    { transform: rotateX(90deg) translateZ(50px); }
.cube-face.bottom { transform: rotateX(-90deg) translateZ(50px); }
</style>
<div class="cube-container">
  <div class="cube">
    <div class="cube-face front">AI</div>
    <div class="cube-face back">DS</div>
    <div class="cube-face right">ML</div>
    <div class="cube-face left">NLP</div>
    <div class="cube-face top">CV</div>
    <div class="cube-face bottom">DL</div>
  </div>
</div>
'''

st.markdown(cube_html, unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; font-family:Segoe UI, sans-serif;'>🤖 AI Software Suite</h1>
<p style='text-align:center; color:var(--text-color);'>A modern platform for Audio Deepfake Detection and Software Defect Prediction</p>
""", unsafe_allow_html=True)

if page == "Audio Deepfake Detection":
    # --- Audio Deepfake Detection UI ---
    import os
    from glob import glob
    st.header("🔊 Audio Deepfake Detection")
    st.write("Select one or more audio files and one or more models to predict if each audio is Bonafide or Deepfake.")
    with st.expander("How does it work?", expanded=False):
        st.info("""
        - Select one or more audio files from the dataset folder.
        - You can search for audio files by name.
        - Choose one or more models: SVM, Logistic Regression, Perceptron, or DNN.
        - The system extracts MFCC features and predicts authenticity for each file with each model.
        """)
    st.markdown("---")
    # Recursively find all .wav files in deepfake_detection_dataset_urdu
    audio_root = "deepfake_detection_dataset_urdu"
    all_audio_files = [os.path.relpath(f, audio_root) for f in glob(os.path.join(audio_root, "**", "*.wav"), recursive=True)]
    # Add urdu_audio_100 files as well
    urdu_audio_dir = "urdu_audio_100"
    urdu_audio_files = [os.path.join(urdu_audio_dir, f) for f in os.listdir(urdu_audio_dir) if f.endswith('.wav')]
    all_audio_files += urdu_audio_files
    all_audio_files = sorted(all_audio_files)
    # Search box for filtering audio files
    search_query = st.text_input("Search audio files by name (case-insensitive):", "")
    filtered_audio_files = [f for f in all_audio_files if search_query.lower() in os.path.basename(f).lower()]
    selected_files = st.multiselect("Select Audio File(s)", filtered_audio_files, default=filtered_audio_files[:1], key="audio_file_select")
    model_choices_audio = st.multiselect("Select Model(s)", ["SVM", "Logistic Regression", "Perceptron", "DNN"], default=["SVM"], key="audio_model_select")
    if st.button("Predict Audio", help="Run prediction on selected audio files."):
        if selected_files and model_choices_audio:
            if len(selected_files) == 1 and len(model_choices_audio) == 1:
                # Single file, single model
                file = selected_files[0]
                if file.startswith(urdu_audio_dir):
                    file_path = file
                else:
                    file_path = os.path.join(audio_root, file)
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()
                pred_label, confidence = predict_audio_deepfake(audio_bytes, model_choices_audio[0])
                st.success(f"Prediction: {pred_label} (Confidence: {confidence:.2f}) for {os.path.basename(selected_files[0])}")
            else:
                # Multiple files and/or models: show table (rows=audio, cols=models)
                import numpy as np
                results = []
                for file in selected_files:
                    row = {"Audio File": os.path.basename(file)}
                    if file.startswith(urdu_audio_dir):
                        file_path = file
                    else:
                        file_path = os.path.join(audio_root, file)
                    try:
                        with open(file_path, "rb") as f:
                            audio_bytes = f.read()
                        for model in model_choices_audio:
                            pred_label, confidence = predict_audio_deepfake(audio_bytes, model)
                            row[f"{model} Prediction"] = pred_label
                            row[f"{model} Confidence"] = np.round(confidence, 2)
                    except FileNotFoundError:
                        for model in model_choices_audio:
                            row[f"{model} Prediction"] = "File Not Found"
                            row[f"{model} Confidence"] = None
                    results.append(row)
                st.dataframe(pd.DataFrame(results))
        else:
            st.warning("Please select at least one audio file and one model.")
    st.caption("Audio Deepfake Detection | Powered by ML & Signal Processing")

elif page == "Software Defect Prediction":
    st.header("🐞 Software Defect Multi-Label Prediction")
    st.write("Paste a bug report below to predict possible defect types. Select your preferred model(s) for prediction.")
    with st.expander("How does it work?", expanded=False):
        st.info("""
        - Paste a bug report or software issue description.
        - Choose one or more models: SVM, Logistic Regression, Perceptron, or DNN.
        - The system predicts multiple defect types (multi-label) using your chosen model(s).
        """)
    st.markdown("---")
    user_report = st.text_area("Paste or type a bug report here:", help="Enter a detailed bug report or software issue.")
    model_choices = st.multiselect("Select Model(s)", ["SVM", "Logistic Regression", "Perceptron", "DNN"], default=["SVM"], key="defect_model_select")
    if st.button("Predict", help="Predict defect types for the given report."):
        if user_report.strip() == "" or not model_choices:
            st.warning("Please enter a bug report and select at least one model.")
        else:
            X_user = vectorizer.transform([user_report])
            if len(model_choices) == 1:
                model_choice = model_choices[0]
                if model_choice == "SVM":
                    pred = svm.predict(X_user)
                    try:
                        scores = 1 / (1 + np.exp(-svm.decision_function(X_user)))
                    except Exception:
                        scores = pred
                elif model_choice == "Logistic Regression":
                    pred = logreg.predict(X_user)
                    scores = logreg.predict_proba(X_user)
                    scores = np.array([s[:,1] for s in scores]).T
                elif model_choice == "Perceptron":
                    pred = perc.predict(X_user)
                    scores = pred
                else:  # DNN
                    pred = (dnn.predict(X_user.toarray()) > 0.5).astype(int)
                    scores = dnn.predict(X_user.toarray())
                st.subheader("Predicted Labels:")
                results = []
                for i, col in enumerate(label_cols):
                    results.append({
                        "Label": col,
                        "Predicted": "Yes" if pred[0][i] else "No",
                        "Confidence": float(scores[0][i])
                    })
                st.dataframe(pd.DataFrame(results))
            else:
                all_results = []
                for model_choice in model_choices:
                    if model_choice == "SVM":
                        pred = svm.predict(X_user)
                        try:
                            scores = 1 / (1 + np.exp(-svm.decision_function(X_user)))
                        except Exception:
                            scores = pred
                    elif model_choice == "Logistic Regression":
                        pred = logreg.predict(X_user)
                        scores = logreg.predict_proba(X_user)
                        scores = np.array([s[:,1] for s in scores]).T
                    elif model_choice == "Perceptron":
                        pred = perc.predict(X_user)
                        scores = pred
                    else:  # DNN
                        pred = (dnn.predict(X_user.toarray()) > 0.5).astype(int)
                        scores = dnn.predict(X_user.toarray())
                    for i, col in enumerate(label_cols):
                        all_results.append({
                            "Model": model_choice,
                            "Label": col,
                            "Predicted": "Yes" if pred[0][i] else "No",
                            "Confidence": float(scores[0][i])
                        })
                st.dataframe(pd.DataFrame(all_results))
    st.caption("Software Defect Prediction | Multi-label ML | Data Science 2025")
