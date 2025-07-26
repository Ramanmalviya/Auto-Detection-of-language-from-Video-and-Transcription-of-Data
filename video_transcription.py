import os
import glob
import moviepy.editor as mp
import torch
import torchaudio
from transformers import AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC

# Load AI4Bharat Indic Conformer model ONCE globally for efficiency
MODEL_PATH = "ai4bharat/indic-conformer-600m-multilingual"
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Supported languages and their codes including Punjabi
LANG_CODES = {
    "english": "en",
    "hindi": "hi",
    "gujarati": "gu",
    "marathi": "mr",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "punjabi": "pa",
}

def video_to_audio(input_video_path, output_audio_path):
    video = mp.VideoFileClip(input_video_path)
    video.audio.write_audiofile(output_audio_path, logger=None)
    print(f"[{os.path.basename(input_video_path)}] Audio saved as: {output_audio_path}")

def load_audio(audio_path):
    wav, sr = torchaudio.load(audio_path)
    wav = torch.mean(wav, dim=0, keepdim=True)  # Mono
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)
    return wav

def detect_language(audio_file):
    """
    Dummy language detector replacement - 
    since AI4Bharat model doesn't output detected language directly,
    you might have to build or plug in a separate language ID or 
    assume you know language from metadata or filename.
    Here we fallback to 'hindi' or can customize.
    """
    # For demonstration, try default or implement your own detection logic here.
    print(f"[{os.path.basename(audio_file)}] Using default fallback language 'hindi'")
    return "hi"

def transcribe_audio(audio_file, language_code, output_folder):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    language_code = language_code.lower()

    # Use specialized Wav2Vec2 model for Punjabi as in your original code
    if language_code == "pa":
        try:
            processor = Wav2Vec2Processor.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")
            wav2vec_model = Wav2Vec2ForCTC.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")
            speech_array, sampling_rate = torchaudio.load(audio_file)
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                speech_array = resampler(speech_array)
            speech = speech_array.squeeze().numpy()
            inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = wav2vec_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            out_path = os.path.join(output_folder, base_name + "_punjabi.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"[{base_name}] Punjabi transcription saved to: {out_path}")
            return transcription
        except Exception as e:
            print(f"[{base_name}] Punjabi transcription failed: {e}")
            return None

    # For other supported languages, use AI4Bharat IndicConformer model
    elif language_code in LANG_CODES:
        try:
            wav = load_audio(audio_file)
            transcription = model(wav, language_code, "ctc")
            out_path = os.path.join(output_folder, base_name + f"_{language_code}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"[{base_name}] Transcription ({language_code}) saved to: {out_path}")
            print("Preview:")
            print(transcription)
            return transcription
        except Exception as e:
            print(f"[{base_name}] AI4Bharat transcription failed: {e}")
            return None

    # Fallback: Language not supported here, could extend with whisperx or others if needed
    else:
        print(f"[{base_name}] Unsupported language code '{language_code}' for transcription.")
        return None

def process_videos_and_transcribe(video_folder, audio_folder, output_folder):
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv',
                        '.MP4', '.AVI', '.MOV', '.MKV', '.FLV']

    video_files = []
    for ext in video_extensions:
        video_files += glob.glob(os.path.join(video_folder, f'*{ext}'))
    print(f"Found {len(video_files)} video files in {video_folder}")

    for video_file in video_files:
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        audio_file = os.path.join(audio_folder, f"{base_name}.wav")
        try:
            video_to_audio(video_file, audio_file)
            lang_code = detect_language(audio_file)  # Here replaced to simplified or fallback detection
            transcribe_audio(audio_file, lang_code, output_folder)
        except Exception as e:
            print(f"[{base_name}] Processing failed: {e}")

# -------------------- USAGE ---------------------
video_folder = "/content/video_folder"
audio_folder = "/content/audio_extracted"
output_folder = "/content/transcripts"

process_videos_and_transcribe(video_folder, audio_folder, output_folder)
