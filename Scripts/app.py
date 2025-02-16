import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['SPEECHBRAIN_CACHE'] = os.path.join(BASE_DIR, 'Models', 'speechbrain')

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import os
import sys
import streamlit as st
import subprocess
import tempfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio(language, audio_file_path):
    script_map = {
        "Bengali": "Scripts/Bengali_Speech.py",
        "Hindi": "Scripts/hindi_speech.py",
        "Punjabi": "Scripts/punjabi_speech.py",
        "Tamil": "Scripts/tamil_speech.py"
    }
    
    if language not in script_map:
        return "Error: Unsupported language."

    script_path = os.path.join(os.getcwd(), script_map[language])
    
    try:
        result = subprocess.run(
            [sys.executable, script_path, audio_file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logging.error(f"Error processing audio: {result.stderr}")
            return f"Error processing audio: {result.stderr}"
        
        return result.stdout
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        return f"Exception occurred: {e}"

st.title("Multilingual Speech Processing")

language = st.selectbox("Select Language", ["Bengali", "Hindi", "Punjabi", "Tamil"])

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if st.button("Process Audio"):
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_path = temp_audio_file.name
            temp_audio_file.write(audio_file.getbuffer())
        
        try:
            output = process_audio(language, temp_audio_path)
            
            st.text_area("Processing Output", output, height=300)
        finally:
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logging.error(f"Error deleting temporary file: {e}")
    else:
        st.error("Please upload an audio file.")
