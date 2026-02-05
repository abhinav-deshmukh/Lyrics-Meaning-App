"""
Step 3: Interpret lyrics using GPT-4
Run this with: python interpret.py "lyrics text here"
Or pipe from transcribe: python transcribe.py song.mp3 | python interpret.py
"""

import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INTERPRETATION_PROMPT = """You are an expert in South Asian music, poetry, and culture. 
A user has shared song lyrics that were transcribed from audio (there may be some transcription errors).

Your task:
1. **Identify the language** (Urdu, Punjabi, Hindi, or mixed)
2. **Translate to English** - preserve the poetic structure and emotional tone, not just literal meaning
3. **Interpret the deeper meaning** - explain metaphors, cultural references, and emotional themes
4. **Note any famous references** - if this is a well-known song, mention the artists and context

Format your response as:

## Language
[identified language]

## English Translation
[poetic translation, maintaining verse structure]

## Meaning & Cultural Context
[2-3 paragraphs explaining the deeper meaning, metaphors, and cultural significance]

## Notable References
[any cultural, literary, or historical references in the lyrics]

---
Here are the lyrics to interpret:

{lyrics}
"""

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
    # Get lyrics from argument or stdin
    if len(sys.argv) > 1:
        lyrics = " ".join(sys.argv[1:])
    elif not sys.stdin.isatty():
        lyrics = sys.stdin.read()
    else:
        print("Usage: python interpret.py \"lyrics text\"")
        print("   or: cat lyrics.txt | python interpret.py")
        sys.exit(1)
    
    if not lyrics.strip():
        print("Error: No lyrics provided")
        sys.exit(1)
    
    print("Interpreting lyrics...")
    print("=" * 50)
    
    interpretation = interpret_lyrics(lyrics)
    print(interpretation)

if __name__ == "__main__":
    main()
