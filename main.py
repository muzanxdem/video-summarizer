import streamlit as st
import streamlit as st
import yt_dlp
import whisper
import os
import requests

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def download_youtube_audio(url, output_path="audio.mp3"):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "outtmpl": output_path,
        "quiet": False,  # Enable logging to see yt-dlp debug messages
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Audio file not created: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {e}")


def transcribe_audio(model, audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text_ollama(prompt, model="llama3.2"):
    url = f"http://192.168.100.140:11434"
    payload = {
        "model": model,
        "prompt": prompt,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "No response from Ollama.")
    else:
        return f"Error: {response.status_code}, {response.text}"

# Streamlit app
def main():
    st.title("YouTube Video Summarizer (with Ollama)")
    st.write("Provide a YouTube link, and the app will summarize the video.")

    # User Input
    youtube_url = st.text_input("Enter YouTube URL:")
    if st.button("Summarize"):
        if youtube_url:
            st.info("Downloading audio from YouTube...")
            audio_file = download_youtube_audio(youtube_url)

            st.info("Transcribing audio...")
            whisper_model = load_whisper_model()
            transcription = transcribe_audio(whisper_model, audio_file)
            st.write("**Transcription:**")
            st.text_area("", transcription, height=200)

            st.info("Generating summary with Ollama...")
            summary_prompt = f"Summarize the following text:\n\n{transcription}"
            summary = summarize_text_ollama(summary_prompt)
            st.write("**Summary:**")
            st.success(summary)

            # Clean up downloaded audio
            if os.path.exists(audio_file):
                os.remove(audio_file)
        else:
            st.error("Please provide a valid YouTube URL.")

if __name__ == "__main__":
    main()
