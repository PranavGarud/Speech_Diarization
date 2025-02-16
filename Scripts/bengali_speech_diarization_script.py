import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['SPEECHBRAIN_CACHE'] = os.path.join(BASE_DIR, 'Models', 'speechbrain')

import os
import time
import subprocess
import torch
import numpy as np
import torchaudio
import imageio_ffmpeg
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForCTC, AutoProcessor
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding



def log_with_timestamp(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

def convert_to_wav(input_file, output_wav):
    log_with_timestamp("Starting audio conversion...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", output_wav],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if os.path.exists(output_wav):
            log_with_timestamp(f"Successfully converted to {output_wav}")
            return output_wav
        raise RuntimeError("Conversion failed - output file not created")
    except subprocess.CalledProcessError as e:
        log_with_timestamp(f"FFmpeg error: {str(e)}")
        raise

def perform_diarization(audio_path, num_speakers):
    log_with_timestamp("Starting speaker diarization...")
    os.environ["TORCH_HOME"] = os.path.join(BASE_DIR, "Models", "cache")
    speechbrain_path = os.path.join(BASE_DIR, "Models", "speechbrain", "spkrec-ecapa-voxceleb")
    if not os.path.exists(speechbrain_path):
        raise FileNotFoundError(f"SpeechBrain model not found at: {speechbrain_path}")
    embedding_model = PretrainedSpeakerEmbedding(
        speechbrain_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    duration = waveform.shape[1] / 16000  # samples / sample_rate
    segment_duration = 2.0  # Reduced from 3.0
    overlap_duration = 0.5  # Add overlap between segments
    step_duration = segment_duration - overlap_duration
    num_segments = int((duration - overlap_duration) / step_duration)
    segments = []
    for i in range(num_segments):
        start_time = i * step_duration
        end_time = min(start_time + segment_duration, duration)
        segments.append({
            "start": start_time,
            "end": end_time
        })
    if len(segments) < 2:
        segments.append({
            "start": 0,
            "end": min(duration, 1.5)
        })
        segments.append({
            "start": min(duration - 1.5, 1.5),
            "end": duration
        })
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
        start_sample = int(segment["start"] * 16000)
        end_sample = int(segment["end"] * 16000)
        segment_audio = waveform[:, start_sample:end_sample]
        if len(segment_audio.shape) == 1:
            segment_audio = segment_audio.unsqueeze(0).unsqueeze(0)
        elif len(segment_audio.shape) == 2:
            segment_audio = segment_audio.unsqueeze(0)
        with torch.no_grad():
            embeddings[i] = embedding_model(segment_audio)  # Already returns numpy array
    embeddings = np.nan_to_num(embeddings)
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    log_with_timestamp(f"Diarization completed: Assigned speakers to {len(set(labels))} clusters")
    return segments

def process_audio_chunks(audio_path, diarization_segments, model, processor, sampling_rate=16000):
    transcriptions = []
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, sampling_rate)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform - waveform.mean()
    waveform = waveform / (waveform.std() + 1e-7)
    for segment in diarization_segments:
        start_sample = int(segment["start"] * sampling_rate)
        end_sample = int(segment["end"] * sampling_rate)
        segment_audio = waveform[:, start_sample:end_sample].squeeze().numpy()
        if len(segment_audio) < sampling_rate:  # if segment is less than 1 second
            padding = np.zeros(sampling_rate - len(segment_audio))
            segment_audio = np.concatenate([segment_audio, padding])
        inputs = processor(
            segment_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        with torch.no_grad():
            outputs = model(
                input_values=inputs.input_values,
                attention_mask=inputs.attention_mask,
                return_dict=True
            )
            logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            group_tokens=True  # Group repeated tokens
        )[0].strip()
        if not transcription.strip():
            continue
        transcriptions.append({
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment["speaker"],
            "text": transcription
        })
    return transcriptions

def main():
    try:
        audio_path = os.path.join(BASE_DIR, "Audio Files", "output_bengali.wav")  # Corrected path construction
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        log_with_timestamp("Loading ASR model and processor...")
        model_path = os.path.join(BASE_DIR, "Models", "Bengali model", "vakyansh-wav2vec2-bengali-bnm-200")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        model = AutoModelForCTC.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        wav_file = "../converted_audio.wav"
        convert_to_wav(audio_path, wav_file)
        num_speakers = 2
        speaker_segments = perform_diarization(wav_file, num_speakers)
        log_with_timestamp("Starting transcription with speaker labels...")
        transcriptions = process_audio_chunks(wav_file, speaker_segments, model, processor)
        log_with_timestamp("\nFinal Transcription Results:")
        for entry in transcriptions:
            print(f"[{entry['start']:.1f}-{entry['end']:.1f}] {entry['speaker']}: {entry['text']}")
        log_with_timestamp("Processing completed successfully")
    except Exception as e:
        log_with_timestamp(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
