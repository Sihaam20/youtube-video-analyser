import streamlit as st
import pytube
from pytube import YouTube
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
import librosa
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper Model
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer)

# Load translation model and tokenizer
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
def download_and_extract_audio(youtube_url):
    """Downloads YouTube video audio and converts it to WAV format.

    Args:
        youtube_url: The URL of the YouTube video.

    Returns:
        A list of audio file paths (if chunking is used) or None (if error occurs).
    """

    try:
        # Download YouTube video
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        stream.download(filename='ytaudio.mp4')

        # Convert MP4 to WAV using FFmpeg
        import subprocess
        subprocess.run(['ffmpeg', '-i', 'ytaudio.mp4', '-acodec', 'pcm_s16le', '-ar', '16000', 'ytaudio.wav'])

        # Check for potential Out-of-Memory (OOM) error by using chunking
        input_file = 'ytaudio.wav'  # Use the converted WAV file
        sample_rate = librosa.get_samplerate(input_file)

        if sample_rate * librosa.core.frames_to_time(librosa.load(input_file)[0].shape[0]) > 100000000:  # Adjust threshold as needed
            # Audio Chunking
            audio_paths = []
            stream = librosa.stream(input_file, block_length=30, frame_length=16000, hop_length=16000)
            for i, speech in enumerate(stream):
                sf.write(f'{i}.wav', speech, 16000)
                audio_paths.append(f'{i}.wav')
            return audio_paths
        else:
            # No chunking needed
            return ['ytaudio.wav']  # Return a list with single audio path

    except Exception as e:
        st.error(f"Error downloading or processing audio: {e}")
        return None

def transcribe_audio(audio_paths):
    """Performs speech recognition (ASR) on the audio.

    Args:
        audio_paths: A list of audio file paths.

    Returns:
        A string containing the combined transcript from all audio segments.
    """
    try:
        transcripts = [pipe(audio_path)[0]['transcription'] for audio_path in audio_paths]
        full_transcript = ' '.join(transcripts)
        return full_transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None
    
def translate_summary(summary_text, target_language):
    # Translate the summarized text to the target language
    model_inputs = tokenizer(summary_text, return_tensors="pt", padding=True)
    generated_tokens = translation_model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_language]
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text
def main():
    st.title("YouTube Video Summarization with Whisper Model")

    youtube_url = st.text_input("Enter YouTube video URL (or 'q' to quit): ")

    if youtube_url.lower() == 'q':
        st.stop()
        return

    if not youtube_url:
        st.warning("Please enter a YouTube video URL.")
        return

    # Download and process audio
    audio_paths = download_and_extract_audio(youtube_url)
    if not audio_paths:
        return

    # Perform speech recognition
    full_transcript = transcribe_audio(audio_paths)

    # Display transcript
    if full_transcript:
        st.success("**Full Transcript:**")
        st.write(full_transcript)

        # Button for translation
        if st.button("Translate"):
            target_language = st.selectbox("Select Target Language:", ["Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Urdu"])
            if target_language:
                # Translate summary to the selected language
                translated_summary = translate_summary(full_transcript, target_language.lower())
                st.success(f"**Translated Summary ({target_language}):**")
                st.write(translated_summary)

if __name__ == "__main__":
    main()