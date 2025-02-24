"""
title: Chat with YouTube Pipeline using Ollama
author: BrainDrive.ai
date: 2024-10-30
version: 1.2
license: MIT
description: A custom pipeline that performs YouTube video search, retrieves transcripts, generates transcript summaries, conducts Q&A over transcripts, and searches within transcript/video content.
requirements: pydantic==2.7.4, requests, youtube-search==2.1.2, youtube-transcript-api==0.6.2, pytube==15.0.0, langchain==0.3.3, langchain-community==0.3.2, langchain-ollama==0.2.0, langchain-core==0.3.10, langchain-text-splitters==0.3.0, moviepy, openai-whisper
"""

import os
import json
import requests
import subprocess
from typing import List, Sequence
from pydantic import BaseModel, Field
from langchain import hub
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
from moviepy.editor import AudioFileClip
from langchain_community.document_loaders import YoutubeLoader

WHISPER_MODEL = "small"  # Options: "tiny", "base", "small", "medium", "large"

# ========================
# YOUTUBE SEARCH FUNCTION
# ========================
class YoutubeSearchInput(BaseModel):
    query: str = Field(description="Search query for YouTube videos.")

@tool("search_youtube", args_schema=YoutubeSearchInput, return_direct=False)
def search_youtube(query: str) -> str:
    """Search YouTube and return video URLs based on the query."""
    results = YoutubeSearch(query, 10).to_json()
    data = json.loads(results)
    return str([
        "https://www.youtube.com" + video["url_suffix"]
        for video in data["videos"]
    ])

# =================================
# YOUTUBE TRANSCRIPT WITH WHISPER
# =================================
class YoutubeTranscriptInput(BaseModel):
    youtube_url: str = Field(description="A valid YouTube video URL.")

def download_audio(youtube_url: str, save_path: str) -> str:
    """Download YouTube video and extract audio."""
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    if not audio_stream:
        raise Exception("No audio stream found for this video.")
    
    audio_path = os.path.join(save_path, f"{yt.video_id}.mp4")
    audio_stream.download(output_path=save_path, filename=f"{yt.video_id}.mp4")

    return audio_path

def extract_audio(video_path: str) -> str:
    """Extract audio from video file."""
    audio_path = video_path.replace(".mp4", ".wav")
    clip = AudioFileClip(video_path)
    clip.write_audiofile(audio_path, codec='pcm_s16le')
    clip.close()
    return audio_path

def transcribe_audio(audio_path: str) -> str:
    """Use Whisper to transcribe the extracted audio."""
    command = f"whisper {audio_path} --model {WHISPER_MODEL} --output_format txt"
    subprocess.run(command, shell=True, check=True)
    
    transcript_path = audio_path.replace(".wav", ".txt")
    
    with open(transcript_path, "r") as file:
        transcript = file.read().strip()
    
    return transcript

@tool("get_youtube_transcript", args_schema=YoutubeTranscriptInput, return_direct=False)
def get_youtube_transcript(youtube_url: str) -> str:
    """Retrieve YouTube transcript, using Whisper if unavailable."""
    try:
        # Try getting transcript via YouTube API
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False, language=["en"], translation="en")
        return "\n\n".join(map(repr, loader.load()))
    
    except TranscriptsDisabled:
        print("No transcript available. Using Whisper for transcription...")
        
        # Define save path
        save_path = "downloads"
        os.makedirs(save_path, exist_ok=True)

        # Download and process audio
        video_path = download_audio(youtube_url, save_path)
        audio_path = extract_audio(video_path)
        transcript = transcribe_audio(audio_path)
        
        return transcript

# ========================
# PIPELINE CLASS
# ========================
class Pipeline:
    class Valves(BaseModel):
        OLLAMA_API_BASE_URL: str = "http://localhost:11434" # or http://host.docker.internal:11434
        OLLAMA_API_KEY: str = "if you are hosting ollama, put API key here"
        OLLAMA_API_MODEL: str = "llama3"
        OLLAMA_API_TEMPERATURE: float = 0.7
        AGENT_SYSTEM_PROMPT: str = (
            "You are a smart assistant that searches for YouTube videos, retrieves their transcripts, "
            "analyzes them, and assists users with Q&A over video content."
        )

    def __init__(self):
        self.name = "Chat with YouTube - Ollama"
        self.tools = [search_youtube, get_youtube_transcript]
        self.valves = self.Valves(
            OLLAMA_API_KEY=os.getenv("OLLAMA_API_KEY", "")
        )
        self.pipelines = self.get_openai_models()

    def get_openai_models(self):
        if self.valves.OLLAMA_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OLLAMA_API_KEY}",
                    "Content-Type": "application/json"
                }
                response = requests.get(
                    f"{self.valves.OLLAMA_API_BASE_URL}/models", headers=headers
                )
                models = response.json()
                return [
                    {"id": model["id"], "name": model.get("name", model["id"])}
                    for model in models["data"] if "gpt" in model["id"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [{"id": "error", "name": "Could not fetch models from OpenAI."}]
        else:
            return []

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        try:
            model = ChatOllama(
                api_key=self.valves.OLLAMA_API_KEY,
                model=self.valves.OLLAMA_API_MODEL,
                temperature=self.valves.OLLAMA_API_TEMPERATURE
            )
            tools: Sequence[BaseTool] = self.tools
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.valves.AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])
            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            response = agent_executor.invoke({"input": user_message, "chat_history": messages})
            return response["output"]
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
