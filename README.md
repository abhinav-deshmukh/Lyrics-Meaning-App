# 🎶 Surasa

**सुर + रस — The essence of melody**

An AI-powered app that reveals the true meaning behind songs in any language.

## What it does

1. **Search** for any song or paste a YouTube link
2. **Transcribes** audio with timestamps (Whisper AI)
3. **Translates** lyrics to English (preserving poetic nuance)
4. **Interprets** cultural context, metaphors, and deeper meaning
5. **Karaoke mode** — synced lyrics as the song plays

## Why this exists

Songs contain metaphors, cultural references, and emotional depth that gets lost in simple translation. Surasa bridges that gap for music in any language — French chansons, K-pop, Latin hits, Arabic classics, Bollywood, and more.

## Tech stack

- **Transcription**: OpenAI Whisper API (99+ languages, auto-detect)
- **Interpretation**: Claude Sonnet (poetic translation + cultural context)
- **Audio**: yt-dlp (YouTube download)
- **UI**: Streamlit

## Run locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to .env
cp .env.example .env
# Edit .env and add your key

# Run the app
streamlit run app.py
```

## Status

✅ Working prototype
