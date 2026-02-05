"""
Full pipeline: Audio → Transcription → Interpretation
Run with: python lyrics_meaning.py <audio_file>
"""

import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INTERPRETATION_PROMPT = """You are an expert in South Asian music, poetry, and culture. 
A user has shared song lyrics that were transcribed from audio (there may be some transcription errors).

Your task: Break down the lyrics LINE BY LINE in this exact format:

For each meaningful line or couplet, output:

**Original:** [the original lyric in its original script]
**Translation:** [English translation, preserving poetic tone]
**Meaning:** [what this line really means - metaphors, emotions, cultural context]

---

Then at the end, add:

## Song Context
[1-2 sentences: what song this is, who performs it, and the overall theme]

---

Important:
- Group lines into natural couplets/verses (don't split rhyming pairs)
- Skip filler words, repetitions, and "la la la" type sounds
- Focus on the meaningful lyrics
- If you recognize the song, use your knowledge to improve accuracy

Here are the lyrics to interpret:

{lyrics}
"""

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

def interpret_lyrics(lyrics: str) -> str:
    """Use GPT-4 to translate and interpret song lyrics."""
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": INTERPRETATION_PROMPT.format(lyrics=lyrics)
            }
        ]
    )
    
    return response.choices[0].message.content

def main():
    if len(sys.argv) < 2:
        print("🎵 Lyrics Meaning App")
        print("-" * 40)
        print("Usage: python lyrics_meaning.py <audio_file>")
        print("Example: python lyrics_meaning.py song.mp3")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Step 1: Transcribe
    print(f"🎤 Transcribing: {audio_path}")
    print("-" * 40)
    transcript = transcribe_audio(audio_path)
    print("📝 Transcription complete!")
    print()
    print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
    print()
    
    # Step 2: Interpret
    print("-" * 40)
    print("🔮 Interpreting meaning...")
    print("-" * 40)
    interpretation = interpret_lyrics(transcript)
    print(interpretation)

if __name__ == "__main__":
    main()
