import streamlit as st
from yt_dlp import YoutubeDL
import whisper
import os
from langchain_community.llms import Ollama
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Function to download audio from YouTube using yt-dlp
def download_audio(youtube_url):
    options = {
        "format": "bestaudio/best",
        "outtmpl": "audio.%(ext)s",
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return info["requested_downloads"][0]["filepath"]

# Function to preprocess text (remove punctuation and stopwords)
def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.lower().split()  # Convert to lowercase and split into words
    words = [word for word in words if word not in STOPWORDS]  # Remove stopwords
    return words

# Sidebar UI
st.sidebar.title("Intelvisual: YouTube Video Summarizer")
st.sidebar.write("Paste a YouTube URL to transcribe and summarize the video.")
youtube_url = st.sidebar.text_input("YouTube URL", "")

# Main interface
st.title("YouTube Video Summarizer")
st.write("This app transcribes and summarizes YouTube videos for quick insights.")

if st.sidebar.button("Summarize"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL.")
    else:
        try:
            # Step 1: Download audio
            st.write("🔄 Downloading audio...")
            audio_file = download_audio(youtube_url)
            st.success("✅ Audio downloaded successfully!")

            # Step 2: Transcribe audio using Whisper
            st.write("🔄 Transcribing audio...")
            model = whisper.load_model("base")  # Use 'small', 'medium', or 'large' for better accuracy
            transcription = model.transcribe(audio_file)
            transcription_text = transcription["text"]

            st.success("✅ Transcription completed!")
            st.subheader("Transcription")
            st.text_area("Full Transcription", transcription_text, height=300)

            # Step 3: Word Cloud Analysis
            st.write("🔄 Generating word cloud analysis...")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(transcription_text)

            # Display the word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.success("✅ Word cloud generated!")

            # Step 4: Top 10 Keywords (Bar Chart)
            st.write("🔄 Generating top 10 keywords bar chart...")
            words = preprocess_text(transcription_text)
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(10)

            # Data for plotting
            keywords, frequencies = zip(*most_common_words)

            # Plot the bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(keywords, frequencies)
            ax.set_title("Top 10 Keywords")
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Keywords")
            st.pyplot(fig)
            st.success("✅ Top 10 keywords chart generated!")

            # Step 5: Summarize transcription using Ollama's Llama3.2
            st.write("🔄 Summarizing transcription...")
            llm = Ollama(model="llama3.2")  # Specify the exact model

            # Instruction prompt with explicit language requirement
            instruction_prompt = (
                "Provide a comprehensive summary of the provided context information in English, do not use other languages "
                "even if the context is in a different language. The summary should cover all the key points "
                "and main ideas presented in the original text, while also condensing the information into a "
                "concise and easy-to-understand format. Please ensure that the summary includes relevant details "
                "and examples that support the main ideas, while avoiding any unnecessary information or repetition.\n\n"
                f"Context:\n{transcription_text}"
            )

            # Generate summary
            summary = llm(instruction_prompt)

            st.success("✅ Summary completed!")
            st.subheader("Summary")
            st.text_area("Video Summary", summary, height=300)

            # Step 6: Clean up
            os.remove(audio_file)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
