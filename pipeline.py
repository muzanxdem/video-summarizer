import os
import requests
import json
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, Form, UploadFile
from yt_dlp import YoutubeDL
import whisper
import ollama
import shutil
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

app = FastAPI()

class YouTubeInput(BaseModel):
    youtube_url: str = Field(description="A valid YouTube video URL.")

@tool("download_and_transcribe", args_schema=YouTubeInput, return_direct=False)
def download_and_transcribe(youtube_url: str) -> str:
    """Downloads audio from YouTube, transcribes it using Whisper, and returns the transcript."""
    options = {"format": "bestaudio/best", "outtmpl": "audio.%(ext)s"}
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_file = info["requested_downloads"][0]["filepath"]
    
    model = whisper.load_model("base")
    transcription = model.transcribe(audio_file)
    os.remove(audio_file)
    
    return transcription["text"]

class TranscriptInput(BaseModel):
    transcript: str = Field(description="The full transcript of the video.")

@tool("summarize_transcript", args_schema=TranscriptInput, return_direct=False)
def summarize_transcript(transcript: str) -> str:
    """Summarizes a transcript using Ollama's Llama 3.2 model."""
    model = ChatOllama(model="llama3.2", temperature=0.7)
    prompt = f"Summarize the following transcript:\n\n{transcript}"
    response = model(prompt)
    return response["message"]["content"]

class QuestionInput(BaseModel):
    transcript: str = Field(description="The full transcript of the video.")
    question: str = Field(description="User's question about the video content.")

@tool("answer_question", args_schema=QuestionInput, return_direct=False)
def answer_question(transcript: str, question: str) -> str:
    """Answers questions based on the provided transcript."""
    model = ChatOllama(model="llama3.2", temperature=0.7)
    prompt = (
        f"Use the following transcript to answer the question:\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Question: {question}"
    )
    response = model(prompt)
    return response["message"]["content"]

@app.post("/process/youtube")
async def process_youtube(youtube_url: str = Form(...)):
    """Handles YouTube video transcription and summarization."""
    transcript = download_and_transcribe(youtube_url)
    summary = summarize_transcript(transcript)
    return {"transcription": transcript, "summary": summary}

@app.post("/ask")
async def ask_question(transcript: str = Form(...), question: str = Form(...)):
    """Handles Q&A based on the transcript."""
    answer = answer_question(transcript, question)
    return {"answer": answer}

class YouTubePipeline:
    class Config(BaseModel):
        API_BASE_URL: str = "http://localhost:8000"
        MODEL: str = "llama3.2"
        TEMPERATURE: float = 0.7
        SYSTEM_PROMPT: str = "You are an AI assistant for transcribing, summarizing, and answering questions about YouTube videos."

    def __init__(self):
        self.tools = [download_and_transcribe, summarize_transcript, answer_question]
        self.config = self.Config()

    def execute_pipeline(self, user_message: str, messages: List[dict]):
        """Runs the AI agent to process user queries."""
        model = ChatOllama(model=self.config.MODEL, temperature=self.config.TEMPERATURE)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.SYSTEM_PROMPT),
            ("user", "{input}")
        ])
        agent = create_tool_calling_agent(model, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        response = agent_executor.invoke({"input": user_message, "chat_history": messages})
        return response["output"]
