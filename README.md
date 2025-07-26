
# Auto-Detection-of-language-from-Video-and-Transcription-of-Data
Automatically extract, detect language, and transcribe audio from video files


## Used By

This project is used by the following companies:

- CDAC (Pune)


## Features

This entire transcription pipeline runs completely offline on your local machine (or server) after initial setup. All audio extraction, language detection, and speech recognition—including Punjabi and Bengali transcription—are performed locally using pre-trained models without requiring any internet connection during processing. This ensures data privacy and fast, reliable operation without dependence on external APIs or cloud services.
Video to audio extraction (WAV)

Automatic language detection via WhisperX

ASR Transcription:

Punjabi: Wav2Vec2 (HuggingFace model)

Bengali: banglaspeech2text

Other languages: WhisperX

Batch processing of entire folders

Customizable paths for input videos, audios, and output transcripts


## Tech 

Utilizes state-of-the-art ASR models:

WhisperX: For multilingual speech recognition, automatic language detection, and transcription.

Wav2Vec2 (HuggingFace): Specifically fine-tuned for high-accuracy Punjabi transcription.

banglaspeech2text: Deep learning ASR focused on Bengali language support.

Employs efficient Python libraries for media processing:

moviepy: To extract audio from video files.

torchaudio & torch: For audio loading, resampling, and passing data to ASR models.

Provides an automated workflow to:

Batch-convert videos to audio,

Detect spoken language,

Transcribe speech—handling multiple languages seamlessly and accurately.

