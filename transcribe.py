"""
Step 2: Test Whisper transcription
Run this with: python transcribe.py <path_to_audio_file>
"""

import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI's Whisper API."""
    client = OpenAI()
    
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    
    return transcript

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        print("Example: python transcribe.py test_audio/song.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    print(f"Transcribing: {audio_path}")
    print("-" * 40)
    
    transcript = transcribe_audio(audio_path)
    
    print("Transcription:")
    print(transcript)
    print("-" * 40)
    print("Done!")

if __name__ == "__main__":
    main()
