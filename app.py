import streamlit as st
from yt_dlp import YoutubeDL
import whisper
import os
from langchain_community.llms import Ollama
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

# Initialize session state for results
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

# Define custom stopwords in the backend
custom_stopwords = [
    "yeah", "uh", "um", "okay", "well", "you know", "actually", "like", "basically", "ini", "itu"
]

# Function to download audio from YouTube using yt-dlp
def download_audio(youtube_url):
    options = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return info["requested_downloads"][0]["filepath"]

# Function to extract top keywords with custom stopwords support
def extract_top_keywords(text, n=10):
    stop_words = set(stopwords.words("english"))
    stop_words.update(custom_stopwords)  # Merge with custom stopwords
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(n)

# Sidebar UI
st.sidebar.title("Intelvisual: YouTube Video Summarizer")
st.sidebar.write("Upload a video file or paste a YouTube URL to transcribe, summarize, and ask questions.")

# Input options
input_type = st.sidebar.radio("Select input type:", ["YouTube URL", "Video File"])
youtube_url = None
uploaded_file = None

if input_type == "YouTube URL":
    youtube_url = st.sidebar.text_input("YouTube URL", "")
elif input_type == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mkv", "avi"])

# Main interface
st.title("YouTube Video Summarizer and Q&A")
st.write("This app transcribes, summarizes, and enables you to ask questions about YouTube videos or uploaded video files.")

if st.sidebar.button("Summarize"):
    if input_type == "YouTube URL" and not youtube_url:
        st.error("Please enter a valid YouTube URL.")
    elif input_type == "Video File" and not uploaded_file:
        st.error("Please upload a valid video file.")
    else:
        try:
            # Step 1: Get audio file
            if input_type == "YouTube URL":
                st.write("🔄 Downloading audio from YouTube...")
                audio_file = download_audio(youtube_url)
                st.success("✅ Audio downloaded successfully!")
            elif input_type == "Video File":
                st.write("🔄 Extracting audio from uploaded file...")
                video_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
                audio_file = "uploaded_audio.wav"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Convert video to audio using ffmpeg
                os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_file}")
                os.remove(video_path)
                st.success("✅ Audio extracted successfully!")

            # Step 2: Transcribe audio using Whisper
            st.write("🔄 Transcribing audio...")
            model = whisper.load_model("base")  # Use 'small', 'medium', or 'large' for better accuracy
            transcription = model.transcribe(audio_file)
            st.session_state.transcription_text = transcription["text"]

            st.success("✅ Transcription completed!")
            st.subheader("Transcription")
            st.text_area("Full Transcription", st.session_state.transcription_text, height=300)

            # Step 3: Word Cloud Analysis
            st.write("🔄 Generating word cloud analysis...")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(st.session_state.transcription_text)

            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.success("✅ Word cloud generated!")

            # Step 4: Top 10 Keywords Bar Chart
            st.write("🔄 Analyzing top keywords...")
            top_keywords = extract_top_keywords(st.session_state.transcription_text, n=10)
            keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "Frequency"])

            # Plot bar chart
            st.subheader("Top 10 Keywords")
            st.bar_chart(keywords_df.set_index("Keyword"))

            # Step 5: Summarize transcription using Ollama's Llama3.2
            st.write("🔄 Summarizing transcription...")
            llm = Ollama(model="llama3.2")  # Specify the exact model

            instruction_prompt = (
                "Provide a comprehensive summary of the provided context information in English, do not use other languages "
                "even if the context is in a different language. The summary should cover all the key points "
                "and main ideas presented in the original text, while also condensing the information into a "
                "concise and easy-to-understand format.\n\n"
                f"Context:\n{st.session_state.transcription_text}"
            )
            st.session_state.summary = llm(instruction_prompt)

            st.success("✅ Summary completed!")
            st.subheader("Summary")
            st.text_area("Video Summary", st.session_state.summary, height=300)

            # Step 6: Clean up
            os.remove(audio_file)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Question-Answering Section
if st.session_state.transcription_text:
    st.subheader("Ask Questions About the Transcript")
    question = st.chat_input("Enter your question:")

    if question:
        st.write("🔄 Generating an answer...")
        try:
            llm = Ollama(model="llama3.2")  # Ensure the LLM is initialized
            context_prompt = (
                "Answer the following question based on the provided transcript. If the answer is not available, respond with 'Information not found.'\n\n"
                f"Transcript:\n{st.session_state.transcription_text}\n\n"
                f"Question: {question}"
            )
            st.session_state.answer = llm(context_prompt)
            st.success("✅ Answer generated!")
            st.write("**Answer:**", st.session_state.answer)
        except Exception as e:
            st.error(f"An error occurred while processing the question: {str(e)}")
