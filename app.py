import streamlit as st
import whisper
import moviepy.editor as mp
import os
import tempfile

# Load the Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("large")

model = load_model()

# Streamlit app title and description
st.title("Audio/Video Transcription & Translation")
st.write("Upload a video or audio file, and choose to transcribe or translate it using Whisper large model.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio or video file", type=["mp4", "mp3", "wav", "m4a"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Convert video to audio if needed
    if uploaded_file.name.endswith(".mp4"):
        st.info("Extracting audio from video...")
        video = mp.VideoFileClip(temp_file_path)
        audio_path = temp_file_path.replace(".mp4", ".mp3")
        video.audio.write_audiofile(audio_path)
        temp_file_path = audio_path

    # Option to transcribe or translate
    option = st.radio("Choose an option", ["Transcribe", "Translate"])

    # Perform transcription or translation
    if st.button("Process"):
        st.info("Processing the file...")
        if option == "Transcribe":
            result = model.transcribe(temp_file_path)
        elif option == "Translate":
            result = model.transcribe(temp_file_path, task="translate")

        # Display the result
        st.subheader("Output")
        st.text_area("Result", result["text"], height=300)

        # Option to download the result
        st.download_button("Download Result", result["text"], file_name="output.txt")

    # Clean up temporary files
    os.remove(temp_file_path)
