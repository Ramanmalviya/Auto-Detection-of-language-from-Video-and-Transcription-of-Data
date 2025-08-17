import os
import time
import torch
import ffmpeg
import whisperx
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Step 1: Extract audio from video --------------------
def extract_audio(video_path, audio_path):
    """
    Take a video file and save its audio as a .wav file (mono, 16kHz).
    """
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar="16000").run(
            overwrite_output=True, quiet=True
        )
        print(f"✅ Audio saved at {audio_path}")
        return audio_path
    except Exception as e:
        print("❌ Failed to extract audio:", e)
        return None


# -------------------- Step 2: Transcribe audio --------------------
def transcribe_audio(audio_path, model_size="medium"):
    """
    Transcribe audio into text.
    - Uses WhisperX for general languages
    - Uses Punjabi or Bengali models if detected
    """
    model = whisperx.load_model(model_size, device=DEVICE, compute_type="float32")
    result = model.transcribe(audio_path)
    lang = result["language"]
    print(f"🌍 Language detected: {lang}")

    # Special case: Punjabi
    if lang == "pa":
        try:
            print("🔤 Using Punjabi Wav2Vec2 model...")
            processor = Wav2Vec2Processor.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")
            wav2vec_model = Wav2Vec2ForCTC.from_pretrained("manandey/wav2vec2-large-xlsr-punjabi")

            speech_array, sr = torchaudio.load(audio_path)
            if sr != 16000:  # resample if needed
                speech_array = torchaudio.transforms.Resample(sr, 16000)(speech_array)

            inputs = processor(speech_array.squeeze().numpy(),
                               sampling_rate=16000,
                               return_tensors="pt",
                               padding=True)
            with torch.no_grad():
                logits = wav2vec_model(inputs.input_values, attention_mask=inputs.attention_mask).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            return transcription, lang
        except Exception as e:
            print("❌ Punjabi transcription failed:", e)
            return None, lang

    # Special case: Bengali
    if lang == "bn":
        try:
            from banglaspeech2text import Speech2Text
            print("🔤 Using Bengali speech-to-text model...")
            stt = Speech2Text("base")
            transcription = stt.recognize(audio_path)
            return transcription, lang
        except Exception as e:
            print("❌ Bengali transcription failed:", e)
            return None, lang

    # For other languages (English, etc.)
    try:
        align_langs = {"en", "fr", "de", "es", "it", "pt", "nl"}
        if lang in align_langs:
            model_a, meta = whisperx.load_align_model(language_code=lang, device=DEVICE)
            aligned_result = whisperx.align(result["segments"], model_a, meta, audio_path, device=DEVICE)
            return aligned_result["segments"], lang
        else:
            # if word-level alignment is not supported
            return result["segments"], lang
    except:
        return result["segments"], lang


# -------------------- Step 3: Save transcript --------------------
def save_transcript(segments, out_path):
    """
    Save transcript to a text file.
    Supports both raw strings and a list of segments.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        if isinstance(segments, str):
            f.write(segments + "\n")
        else:
            for seg in segments:
                f.write(seg["text"] + "\n")
    print(f"📄 Transcript saved: {out_path}")


# -------------------- Step 4: Run on multiple videos --------------------
def process_videos(video_folder, output_folder, model_size="medium"):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(video_folder):
        if file_name.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
            print(f"\n▶️ Processing {file_name} ...")

            # paths
            video_path = os.path.join(video_folder, file_name)
            audio_path = os.path.join(output_folder, file_name.rsplit(".", 1)[0] + "_audio.wav")
            transcript_path = os.path.join(output_folder, file_name.rsplit(".", 1) + "_transcript.txt")

            start = time.time()

            # Step 1: Extract audio
            if extract_audio(video_path, audio_path) is None:
                continue  # skip if audio extraction fails

            # Step 2: Transcribe
            segments, lang = transcribe_audio(audio_path, model_size)

            # Step 3: Save transcript
            transcript_path = transcript_path.replace(".txt", f"_{lang}.txt")
            save_transcript(segments, transcript_path)

            end = time.time()
            print(f"✅ Done in {end - start:.2f} seconds")


# -------------------- Main --------------------
if __name__ == "__main__":
    # Define folders
    video_folder = "/content/drive/MyDrive/CDAC Project/videos"
    output_folder = "/content/transcripts"

    # Run transcription
    process_videos(video_folder, output_folder, model_size="medium")
