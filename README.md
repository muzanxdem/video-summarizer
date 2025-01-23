# YouTube Video Summarizer and Q&A
## Overview
This app provides a seamless interface for users to transcribe, summarize, analyze, and interact with the content of YouTube videos or uploaded video files. It is built using Streamlit and integrates several advanced tools to offer a comprehensive analysis of video content.
## Features
- Input Options:
  - Paste a YouTube URL or upload video files (MP4, MKV, AVI).
- Transcription:
  - Converts audio to text using OpenAI's Whisper model for high-accuracy transcription.
- Keyword Analysis:
  - Extracts the top 10 keywords from the transcript.
  - Visualizes keyword frequencies with bar charts.
- Word Cloud:
  - Generates a word cloud to highlight prominent words in the transcription.
- Summarization:
  - Provides a concise summary of the video content using the Ollama Llama3.2 model.
- Q&A:
  - Allows users to ask specific questions about the transcript.
  - Delivers context-aware answers using advanced language models.
- Custom Stopwords (Backend):
  - Filters out common and custom stopwords to enhance keyword analysis.
 
## Installation
1. Clone the repository.
```
git clone https://github.com/muzanxdem/video-summarizer.git
cd video-summarizer
```
2. Install Dependencies
Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. Install the required Python libraries:
```
pip3 install -r requirements.txt
```
## Usage
1. Run the Streamlit app
```
streamlit run app.py
```
2. You can choose to insert an YouTube URL or a video/audio files as an input.
![image](https://github.com/user-attachments/assets/026b2830-abae-4fc7-815f-29dd7cd05f74)

3. You also can ask a question about the transcript
![image](https://github.com/user-attachments/assets/716f2c70-9885-4c5f-972e-db868099bff0)

## License
This project is licensed under the MIT License. See the [LICENSE](opensource.org) file for details.

