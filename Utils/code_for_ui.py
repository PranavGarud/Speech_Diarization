import os
import sys
import gradio as gr
import subprocess
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

# Set up base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Create FastAPI app
app = FastAPI()

def process_audio(audio_path, language, num_speakers, progress=gr.Progress()):
    """Process audio file with selected language script"""
    try:
        # Map languages to their script paths
        script_map = {
            "Bengali": "Scripts/Bengali_Speech.py",
            "Bengali (Fine-tuned)": "Scripts/Bengali_Speech_FT.py",
            "Hindi": "Scripts/hindi_speech.py",
            "Punjabi": "Scripts/punjabi_speech.py",
            "Tamil": "Scripts/tamil_speech.py"
        }
        
        if language not in script_map:
            return f"Error: Unsupported language {language}"
        
        # Save uploaded audio to temp location
        temp_audio = os.path.join(BASE_DIR, "Audio Files", "input_audio.wav")
        os.makedirs(os.path.dirname(temp_audio), exist_ok=True)
        
        if isinstance(audio_path, str):
            import shutil
            shutil.copy2(audio_path, temp_audio)
        else:
            audio_path.save(temp_audio)
        
        # Update progress
        progress(0.25)  # 25% after saving audio
        
        # Run the appropriate script
        script_path = os.path.join(BASE_DIR, script_map[language])
        result = subprocess.run(
            [sys.executable, script_path, "--speakers", str(num_speakers)],
            capture_output=True,
            text=True
        )
        
        # Update progress
        progress(0.75)  # 75% after running script
        
        if result.returncode != 0:
            return f"Error processing audio: {result.stderr}"
        
        # Final update
        progress(1.0)  # 100% when done
        return result.stdout
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
def create_ui():
    with gr.Blocks(title="Speech Processing UI") as app:
        gr.Markdown("# Multilingual Speech Processing")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath"
                )
                language = gr.Dropdown(
                    choices=[
                        "Bengali",
                        "Bengali (Fine-tuned)",
                        "Hindi",
                        "Punjabi",
                        "Tamil"
                    ],
                    label="Select Language",
                    value="Hindi"
                )
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Number of Speakers"
                )
                process_btn = gr.Button("Process Audio")
            
            with gr.Column():
                output_text = gr.TextArea(
                    label="Processing Output",
                    interactive=False
                )
        
        process_btn.click(
            fn=process_audio,
            inputs=[audio_input, language, num_speakers],
            outputs=output_text,
            show_progress=True  # Show loading spinner
        )
    
    return app

# Create Gradio interface
gr_interface = create_ui()

# Mount Gradio app to FastAPI
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return gr_interface.launch(share=False, inline=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 
