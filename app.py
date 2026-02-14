"""
Surasa - The essence of melody
Discover the deeper meaning behind songs in any language
Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import subprocess
import os
import json
import base64
import hashlib
import html
import urllib.request
import urllib.parse
import time as time_module
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from streamlit_searchbox import st_searchbox

load_dotenv()

# Animated status messages for long operations (client-side JS animation)
def create_animated_status_html(messages, interval_ms=2000):
    """
    Create HTML/JS that animates through status messages client-side.
    This works even when Python is blocked on an API call.
    """
    messages_js = json.dumps(messages)
    return f"""
    <div id="animated-status" style="
        font-size: 14px;
        color: #666;
        padding: 8px 0;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    ">
        <span id="status-icon">💭</span>
        <span id="status-text">{messages[0]}</span>
    </div>
    <script>
        (function() {{
            const messages = {messages_js};
            const textEl = document.getElementById('status-text');
            let idx = 0;
            
            setInterval(() => {{
                idx = (idx + 1) % messages.length;
                if (textEl) {{
                    textEl.style.opacity = 0;
                    setTimeout(() => {{
                        textEl.textContent = messages[idx];
                        textEl.style.opacity = 1;
                    }}, 150);
                }}
            }}, {interval_ms});
        }})();
    </script>
    <style>
        #status-text {{
            transition: opacity 0.15s ease;
        }}
    </style>
    """

@contextmanager
def animated_status(placeholder, messages, interval=2.0):
    """
    Show rotating status messages while a long operation runs.
    Uses client-side JS so it works even when Python is blocked.
    """
    # Show animated HTML
    html = create_animated_status_html(messages, int(interval * 1000))
    placeholder.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        # Clear the animation when done
        placeholder.empty()

# Simple file-based cache for processed songs
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(url: str, language: str) -> str:
    """Generate cache key from URL and language."""
    return hashlib.md5(f"{url}:{language}".encode()).hexdigest()

def get_cached_result(url: str, language: str) -> dict:
    """Try to get cached result for a song."""
    cache_key = get_cache_key(url, language)
    return get_cached_result_by_key(cache_key)

def get_cached_result_by_key(cache_key: str) -> dict:
    """Load cached result by key (e.g. from History)."""
    if not cache_key or not cache_key.replace('-', '').replace('_', '').isalnum():
        return None
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def _get_youtube_metadata(url: str) -> dict:
    """Get channel, duration string, and duration_seconds from YouTube URL."""
    try:
        result = subprocess.run(
            ["yt-dlp", url, "--dump-json", "--flat-playlist"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip().split('\n')[0])
            duration_sec = data.get('duration')
            if duration_sec is None and data.get('duration_string'):
                # Parse "3:45" or "1:02:30" into seconds
                parts = data.get('duration_string', '').strip().split(':')
                try:
                    if len(parts) == 1:
                        duration_sec = int(parts[0])
                    elif len(parts) == 2:
                        duration_sec = int(parts[0]) * 60 + int(parts[1])
                    else:
                        duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                except (ValueError, IndexError):
                    duration_sec = None
            return {
                'channel': data.get('channel', data.get('uploader', 'Unknown')),
                'duration': data.get('duration_string', ''),
                'duration_seconds': duration_sec,
            }
    except Exception:
        pass
    return {'channel': 'Unknown', 'duration': '', 'duration_seconds': None}


# Max duration for suggested/search results (songs only, no long mixes)
MAX_SUGGESTION_DURATION_SEC = 600  # 10 minutes


def _parse_duration_to_seconds(duration_val) -> Optional[int]:
    """Parse duration from int (seconds) or string like '3:45' / '1:02:30'. Returns seconds or None."""
    if duration_val is None:
        return None
    if isinstance(duration_val, (int, float)):
        return int(duration_val) if duration_val >= 0 else None
    s = (duration_val or "").strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return int(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) >= 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def save_to_cache(url: str, language: str, data: dict, title: str = None):
    """Save processed song to cache with metadata for history."""
    import time
    data = dict(data)
    metadata = _get_youtube_metadata(url)
    data['_meta'] = {
        'url': url,
        'title': title or 'Unknown',
        'cached_at': time.strftime('%Y-%m-%d %H:%M'),
        'thumbnail': _youtube_thumbnail_url(_video_id_from_url(url)),
        'channel': metadata.get('channel', 'Unknown'),
        'duration': metadata.get('duration', ''),
        'language': data.get('language'),
        'mood': data.get('mood'),
    }
    cache_key = get_cache_key(url, language)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass  # Fail silently

def get_cached_songs() -> list:
    """List cached songs for History tab (most recent first). Backfill channel/duration from YouTube if missing."""
    songs = []
    try:
        for f in os.listdir(CACHE_DIR):
            if not f.endswith('.json'):
                continue
            path = os.path.join(CACHE_DIR, f)
            try:
                with open(path, 'r') as file:
                    data = json.load(file)
                meta = data.get('_meta', {})
                if not meta:
                    continue
                url = meta.get('url', '')
                channel = meta.get('channel', 'Unknown')
                duration = meta.get('duration', '')
                # Backfill: if missing (e.g. old cache), fetch from YouTube and update cache
                if (not channel or channel == 'Unknown' or not duration) and url:
                    fetched = _get_youtube_metadata(url)
                    if fetched.get('channel') or fetched.get('duration'):
                        channel = fetched.get('channel') or channel
                        duration = fetched.get('duration') or duration
                        meta['channel'] = channel
                        meta['duration'] = duration
                        data['_meta'] = meta
                        try:
                            with open(path, 'w') as out:
                                json.dump(data, out)
                        except Exception:
                            pass
                songs.append({
                    'title': meta.get('title', 'Unknown'),
                    'url': url,
                    'cached_at': meta.get('cached_at', ''),
                    'cache_key': f.replace('.json', ''),
                    'thumbnail': meta.get('thumbnail', ''),
                    'channel': channel,
                    'duration': duration,
                    'language': meta.get('language'),
                    'mood': data.get('mood') or meta.get('mood'),
                })
            except Exception:
                continue
        songs.sort(key=lambda x: x.get('cached_at', ''), reverse=True)
    except Exception:
        pass
    return songs

def get_youtube_suggestions(query: str) -> list:
    """Get autocomplete suggestions from YouTube."""
    if not query or len(query) < 2:
        return []
    
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://suggestqueries.google.com/complete/search?client=youtube&ds=yt&q={encoded_query}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            data = response.read().decode('utf-8')
            # Parse the JSONP response
            # Format: window.google.ac.h(["query",[["suggestion1"],["suggestion2"],...]])
            start = data.find('[[')
            end = data.rfind(']]') + 2
            if start > 0 and end > start:
                suggestions_data = json.loads(data[start:end])
                return [s[0] for s in suggestions_data if s][:8]
    except Exception:
        pass
    
    return []

# Interpretation prompt - optimized for quality translation and cultural context
INTERPRETATION_PROMPT = """You are a language expert helping users understand song lyrics. For each line below, provide:

1. **original**: The exact text as given
2. **romanized**: Phonetic pronunciation in Latin script (if the original uses non-Latin script; otherwise leave empty)
3. **translation**: Natural, poetic English translation that captures the feeling
4. **meaning**: 1-2 sentences explaining cultural context, idioms, wordplay, or emotional subtext

Output ONLY a valid JSON array. No markdown, no commentary.
Format: [{{"original":"...","romanized":"...","translation":"...","meaning":"..."}}]

Lines to interpret:
{segments}
"""

def _youtube_thumbnail_url(video_id: str) -> str:
    """Standard YouTube thumbnail URL (mqdefault = 320x180)."""
    if not video_id:
        return ""
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


def _video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from watch or youtu.be URL."""
    if not url:
        return ""
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    return ""


def search_youtube(query: str, max_results: int = 5) -> list:
    """Search YouTube and return list of results (≤10 min). Fetches extra then filters by duration."""
    try:
        # Request more so we have enough after filtering by 10 min limit
        fetch_count = max(max_results * 3, 15)
        result = subprocess.run(
            ["yt-dlp", f"ytsearch{fetch_count}:{query}", "--dump-json", "--flat-playlist"],
            capture_output=True, text=True, timeout=30
        )
        
        results = []
        for line in result.stdout.strip().split('\n'):
            if line:
                data = json.loads(line)
                duration_sec = data.get('duration')
                if duration_sec is None and data.get('duration_string'):
                    duration_sec = _parse_duration_to_seconds(data.get('duration_string'))
                if duration_sec is None or duration_sec > MAX_SUGGESTION_DURATION_SEC:
                    continue
                vid = data.get('id', '')
                url = f"https://www.youtube.com/watch?v={vid}"
                thumb = data.get('thumbnail') or _youtube_thumbnail_url(vid)
                results.append({
                    'title': data.get('title', 'Unknown'),
                    'url': url,
                    'channel': data.get('channel', data.get('uploader', 'Unknown')),
                    'duration': data.get('duration_string', ''),
                    'thumbnail': thumb,
                })
                if len(results) >= max_results:
                    break
        return results
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def download_audio(url: str, output_dir: str) -> str:
    """Download audio from YouTube URL (optimized for speed)."""
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    
    # Use lower quality (9 = smallest) - we only need it for transcription
    result = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "9",
         "-o", output_template, "--no-playlist", url],
        capture_output=True, text=True, timeout=120
    )
    
    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("audio."):
            return os.path.join(output_dir, f)
    
    raise Exception(f"Download failed: {result.stderr}")

# Chunking: process long audio in pieces to avoid timeouts and improve reliability
CHUNK_DURATION_SEC = 240  # 4 minutes per chunk
CHUNK_OVERLAP_SEC = 0.5   # small overlap to avoid cutting words

def _split_audio_into_chunks(audio_path: str) -> list:
    """
    Split long audio into chunks. Returns list of (chunk_file_path, start_offset_sec).
    If chunking fails or not needed, returns [(audio_path, 0)].
    """
    try:
        from pydub import AudioSegment
        ext = os.path.splitext(audio_path)[1].lstrip('.').lower() or 'mp3'
        audio = AudioSegment.from_file(audio_path, format=ext)
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000.0
        if duration_sec <= CHUNK_DURATION_SEC:
            return [(audio_path, 0.0)]
        chunk_ms = int(CHUNK_DURATION_SEC * 1000)
        overlap_ms = int(CHUNK_OVERLAP_SEC * 1000)
        step_ms = chunk_ms - overlap_ms
        out_dir = os.path.dirname(audio_path)
        chunks = []
        start_ms = 0
        idx = 0
        while start_ms < duration_ms:
            end_ms = min(start_ms + chunk_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join(out_dir, f"chunk_{idx}.{ext}")
            chunk.export(chunk_path, format=ext)
            chunks.append((chunk_path, start_ms / 1000.0))
            start_ms += step_ms
            idx += 1
        return chunks
    except Exception:
        return [(audio_path, 0.0)]

def _transcribe_one_file(audio_path: str, language: str, client) -> list:
    """Single-file transcription (used by transcribe_with_timestamps)."""
    with open(audio_path, "rb") as audio_file:
        params = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"]
        }
        if language and language != "auto":
            params["language"] = language
        transcript = client.audio.transcriptions.create(**params)
    segments = []
    for seg in transcript.segments:
        segments.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text.strip()
        })
    return segments

def transcribe_with_timestamps(audio_path: str, language: str = None) -> list:
    """Transcribe audio with timestamps. Chunks long audio and merges segments; retries if incomplete."""
    import time
    client = OpenAI()
    chunks = _split_audio_into_chunks(audio_path)
    tmp_dir = os.path.dirname(audio_path)
    created_chunk_files = [p for p, _ in chunks if p != audio_path]

    # Retry for connection errors and for incomplete results
    max_retries = 3
    for attempt in range(max_retries):
        try:
            all_segments = []
            for chunk_path, offset_sec in chunks:
                segs = _transcribe_one_file(chunk_path, language, client)
                for s in segs:
                    all_segments.append({
                        'start': s['start'] + offset_sec,
                        'end': s['end'] + offset_sec,
                        'text': s['text']
                    })
            all_segments.sort(key=lambda x: x['start'])

            # Retry if transcription is empty or clearly incomplete
            text_segments = [s for s in all_segments if s['text'].strip()]
            if not text_segments and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            if text_segments and len(' '.join(s['text'] for s in text_segments).strip()) < 20 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue

            # Clean up chunk files we created (not the original)
            for p in created_chunk_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            return all_segments

        except Exception as e:
            for p in created_chunk_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise e


def interpret_segments(segments: list) -> list:
    """
    Single Sonnet call for high-quality interpretation:
    romanization + translation + cultural meaning in one request.
    """
    import time
    client = Anthropic()
    
    # Filter to segments with actual text
    text_segments = [s for s in segments if s['text'].strip()]
    
    if not text_segments:
        for seg in segments:
            seg['romanized'] = ''
            seg['translation'] = '(no lyrics detected)'
            seg['meaning'] = ''
        return segments
    
    # Deduplicate - only interpret unique lyrics
    unique_texts = []
    seen = set()
    for s in text_segments:
        text_lower = s['text'].strip().lower()
        if text_lower not in seen:
            unique_texts.append(s['text'])
            seen.add(text_lower)
    
    segments_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(unique_texts)])
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": INTERPRETATION_PROMPT.format(segments=segments_text)}]
            )
            
            response_text = response.content[0].text
            json_text = response_text
            
            # Remove markdown code blocks if present
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                parts = json_text.split("```")
                if len(parts) >= 2:
                    json_text = parts[1]
            
            json_text = json_text.strip()
            
            # Find the JSON array using bracket matching
            if not json_text.startswith('['):
                start = json_text.find('[')
                if start >= 0:
                    depth = 0
                    end = start
                    for i, c in enumerate(json_text[start:], start):
                        if c == '[':
                            depth += 1
                        elif c == ']':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    json_text = json_text[start:end]
            
            interpretations = json.loads(json_text)

            # Build lookup and apply to segments
            interp_lookup = {}
            for i, text in enumerate(unique_texts):
                if i < len(interpretations):
                    interp_lookup[text.strip().lower()] = interpretations[i]
            
            result = []
            for seg in segments:
                text_key = seg['text'].strip().lower()
                if text_key in interp_lookup:
                    interp = interp_lookup[text_key]
                    seg['romanized'] = interp.get('romanized', '')
                    seg['translation'] = interp.get('translation', '')
                    seg['meaning'] = interp.get('meaning', '')
                else:
                    seg['romanized'] = ''
                    seg['translation'] = ''
                    seg['meaning'] = ''
                result.append(seg)

            # Retry if too many segments have missing translation/meaning
            with_text = [s for s in result if s['text'].strip()]
            complete = sum(1 for s in with_text if (s.get('translation') or '').strip())
            if not with_text or complete >= 0.8 * len(with_text) or attempt == max_retries - 1:
                return result
            time.sleep(1)
            continue

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                for seg in segments:
                    seg['romanized'] = ''
                    seg['translation'] = '(translation failed - please try again)'
                    seg['meaning'] = ''
                return segments
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                for seg in segments:
                    seg['romanized'] = ''
                    seg['translation'] = f'(error: {str(e)[:50]})'
                    seg['meaning'] = ''
                return segments

    return segments


def _is_instrumental_segment(seg: dict) -> bool:
    """True if segment is music/instrumental only (no real lyrics)."""
    import re
    text = (seg.get('text') or '').strip()
    if not text:
        return True
    if not re.search(r'[\w]', text):
        return True
    if text.lower() in {'♪', 'music', 'instrumental', '...', '…', '..'}:
        return True
    if len(text) <= 2 and not text.isalnum():
        return True
    return False


def merge_instrumental_segments(segments: list) -> list:
    """
    Merge consecutive instrumental/music-only segments into one.
    End time of merged intro is extended to the start of the first real lyric
    so the player doesn't show 'music' while singing has already started.
    """
    if not segments:
        return segments
    result = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if not _is_instrumental_segment(seg):
            result.append(dict(seg))
            i += 1
            continue
        run_start = seg['start']
        run_end = seg['end']
        j = i + 1
        while j < len(segments) and _is_instrumental_segment(segments[j]):
            run_end = segments[j]['end']
            j += 1
        next_lyric_start = None
        if j < len(segments):
            next_lyric_start = segments[j]['start']
        merged_end = next_lyric_start if next_lyric_start is not None else run_end
        result.append({
            'start': run_start,
            'end': merged_end,
            'text': '♪',
            'romanized': '(instrumental)',
            'translation': 'Instrumental',
            'meaning': '',
        })
        i = j
    return result


LANGUAGE_MOOD_PROMPT = """You are helping a language-learning app. The user has pasted a short excerpt of text (from audio they are listening to). Identify the language and mood for the app's UI.

Respond with ONLY a JSON object. No other text.
Format: {{"language": "English", "mood": "Upbeat", "summary": "One or two sentences about the theme or feeling."}}

- language: the language of the text, in English (e.g. Korean, Spanish, Japanese, French).
- mood: one word (e.g. Melancholic, Upbeat, Romantic, Peaceful, Energetic, Nostalgic, Joyful, Dreamy, Bittersweet).
- summary: one or two short sentences (under 200 chars) describing theme, story, or feeling.

User's text excerpt:
{excerpt}
"""

def get_language_and_mood(segments: list) -> tuple:
    """Return (language, mood, summary) from Claude based on lyric excerpt. Returns (None, None, None) on failure."""
    text_segments = [s for s in segments if s.get('text', '').strip()][:8]
    if not text_segments:
        return (None, None, None)
    excerpt = "\n".join(s['text'].strip() for s in text_segments)
    try:
        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": LANGUAGE_MOOD_PROMPT.format(excerpt=excerpt)}]
        )
        text = response.content[0].text.strip()
        if '```' in text:
            text = text.split('```')[1].replace('json', '').strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            obj = json.loads(text[start:end])
        else:
            obj = json.loads(text)
        summary = (obj.get('summary') or '').strip() or None
        return (obj.get('language') or None, obj.get('mood') or None, summary)
    except Exception:
        return (None, None, None)

def get_audio_base64(audio_path: str) -> str:
    """Convert audio file to base64 for embedding."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# Mood -> [gradient start, gradient end, wave color] for player theme
MOOD_THEMES = {
    "melancholic": ("#1a1a2e", "#2d1b4e", "rgba(138, 43, 226, 0.15)"),
    "upbeat": ("#2e1a1a", "#4e2d1b", "rgba(255, 180, 50, 0.2)"),
    "romantic": ("#2e1a2a", "#4e1b3d", "rgba(255, 105, 180, 0.2)"),
    "peaceful": ("#1a2e2a", "#1b4e3d", "rgba(0, 200, 150, 0.15)"),
    "energetic": ("#2e1a1a", "#4e2a1a", "rgba(255, 80, 60, 0.2)"),
    "nostalgic": ("#1a252e", "#1b3d4e", "rgba(100, 149, 237, 0.2)"),
    "joyful": ("#2e2a1a", "#4e4a1b", "rgba(255, 215, 0, 0.2)"),
    "dreamy": ("#1e1a2e", "#3d2e4e", "rgba(147, 112, 219, 0.2)"),
    "bittersweet": ("#2a1a2e", "#3d1b4e", "rgba(200, 100, 180, 0.15)"),
}

def create_karaoke_player(audio_base64: str, segments: list, audio_format: str = "mp3", language: str = None, mood: str = None, summary: str = None) -> str:
    """Create HTML/JS karaoke player with optional language/mood badges, summary, and mood theme."""
    import html as htmlmod
    lang_badge = htmlmod.escape(str(language or '—'))
    mood_badge = htmlmod.escape(str(mood or '—'))
    summary_escaped = htmlmod.escape(str(summary or '').strip())
    
    # Convert segments to JSON for JavaScript
    segments_json = json.dumps(segments)
    
    mood_key = (mood or "").strip().lower().replace(" ", "")
    theme = MOOD_THEMES.get(mood_key)
    if not theme:
        for k, v in MOOD_THEMES.items():
            if k in mood_key:
                theme = v
                break
    theme = theme or ("#1a1a2e", "#16213e", "rgba(0, 212, 255, 0.12)")
    bg_start, bg_end, wave_color = theme
    
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        .karaoke-container {{
            font-family: 'Inter', sans-serif;
            max-width: 100%;
            margin: 0 auto;
            background: linear-gradient(135deg, {bg_start} 0%, {bg_end} 100%);
            border-radius: 16px;
            padding: 24px;
            color: white;
            height: 80vh;
            min-height: 400px;
            max-height: 700px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }}
        @media (max-width: 768px) {{
            .karaoke-container {{
                border-radius: 10px;
                padding: 12px;
                height: 85vh;
                max-height: none;
            }}
        }}
        .wave-bars {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            height: 100%;
            width: 100%;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 6px;
            padding: 0 16px;
            pointer-events: none;
            z-index: 9;
        }}
        .wave-bars span {{
            flex: 1;
            max-width: 14px;
            height: 100%;
            min-height: 100%;
            background: {wave_color};
            border-radius: 4px;
            transform-origin: bottom center;
            transform: scaleY(0.15);
            transition: transform 0.05s ease-out;
        }}
        .player-badges {{
            display: flex;
            gap: 6px;
            flex-shrink: 0;
            margin-left: auto;
        }}
        .player-badge {{
            font-size: 0.75em;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(255,255,255,0.15);
            color: rgba(255,255,255,0.95);
            white-space: nowrap;
        }}
        .song-summary {{
            font-size: 0.9em;
            color: rgba(255,255,255,0.85);
            line-height: 1.4;
            margin-bottom: 8px;
            padding: 10px 14px;
            background: {bg_end};
            border-radius: 8px;
            border-left: 3px solid {wave_color.replace('0.2', '0.6').replace('0.15', '0.6').replace('0.12', '0.6')};
            flex-shrink: 0;
            position: sticky;
            top: 72px;
            z-index: 5;
            box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        }}
        .song-summary-label {{
            font-size: 0.7em;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {wave_color.replace('0.2', '0.9').replace('0.15', '0.9').replace('0.12', '0.9')};
            margin-bottom: 4px;
        }}
        
        .audio-controls {{
            position: sticky;
            top: 0;
            z-index: 6;
            flex-shrink: 0;
            margin-bottom: 4px;
            padding: 8px 4px;
            display: flex;
            align-items: center;
            gap: 16px;
            background: {bg_start};
            box-shadow: 0 2px 12px rgba(0,0,0,0.3);
            border-radius: 0 0 8px 8px;
        }}
        .play-pause-btn {{
            width: 52px;
            height: 52px;
            border-radius: 50%;
            border: none;
            background: rgba(0, 212, 255, 0.35);
            color: #fff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            flex-shrink: 0;
            transition: background 0.2s, transform 0.15s;
            box-shadow: 0 2px 12px rgba(0, 212, 255, 0.25);
        }}
        .play-pause-btn:hover {{
            background: rgba(0, 212, 255, 0.55);
            transform: scale(1.05);
        }}
        .play-pause-btn:active {{
            transform: scale(0.98);
        }}
        .audio-controls .audio-wrap {{
            flex: 1;
            min-width: 0;
        }}
        .audio-controls audio {{
            display: none;
        }}
        .seek-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            margin-top: 6px;
        }}
        .seek-row input[type="range"] {{
            flex: 1;
            min-width: 0;
            height: 6px;
            -webkit-appearance: none;
            appearance: none;
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }}
        .seek-row input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #00d4ff;
            cursor: pointer;
        }}
        .seek-row input[type="range"]::-moz-range-thumb {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #00d4ff;
            cursor: pointer;
            border: none;
        }}
        .skip-btn {{
            flex-shrink: 0;
            font-size: 0.75em;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.3);
            background: rgba(255,255,255,0.1);
            color: #ccc;
            cursor: pointer;
        }}
        .skip-btn:hover {{
            background: rgba(255,255,255,0.2);
            color: #fff;
        }}
        
        .lyrics-wrapper {{
            flex: 1;
            min-height: 0;
            position: relative;
            z-index: 2;
            overflow-y: auto;
            scroll-behavior: smooth;
        }}
        .lyrics-container {{
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }}
        .focus-overlay {{
            display: none;
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            align-items: center;
            justify-content: center;
            padding: 40px;
            z-index: 8;
        }}
        .focus-overlay.show {{
            display: flex;
        }}
        .focus-overlay .focus-content {{
            text-align: center;
            max-width: 90%;
        }}
        .focus-overlay .focus-original {{
            font-size: 2em;
            font-weight: 600;
            color: #fff;
            margin-bottom: 12px;
            line-height: 1.4;
        }}
        .focus-overlay .focus-romanized {{
            font-size: 1.4em;
            color: #ffd700;
            font-style: italic;
            margin-bottom: 16px;
        }}
        .focus-overlay .focus-translation {{
            font-size: 1.5em;
            color: #00d4ff;
            font-weight: 500;
            margin-bottom: 12px;
        }}
        .focus-overlay .focus-meaning {{
            font-size: 1em;
            color: #b8b8b8;
        }}
        .focus-overlay .focus-hint {{
            position: absolute;
            bottom: 16px;
            left: 0; right: 0;
            font-size: 0.8em;
            color: #666;
        }}
        
        .lyric-segment {{
            padding: 16px;
            margin: 8px 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            opacity: 0.4;
            border-left: 3px solid transparent;
            cursor: pointer;
        }}
        .lyric-segment:hover {{
            background: rgba(255,255,255,0.06);
            opacity: 0.85;
        }}
        
        .lyric-segment.active {{
            opacity: 1;
            background: rgba(255,255,255,0.1);
            border-left: 3px solid #00d4ff;
            transform: scale(1.02);
        }}
        
        .lyric-segment.past {{
            opacity: 0.6;
        }}
        
        .original {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 4px;
            color: #fff;
        }}
        
        .romanized {{
            font-size: 1.1em;
            color: #ffd700;
            margin-bottom: 8px;
            font-style: italic;
            letter-spacing: 0.5px;
        }}
        
        .translation {{
            font-size: 1.1em;
            color: #00d4ff;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .meaning {{
            font-size: 0.85em;
            color: #b8b8b8;
            padding: 8px 12px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 6px;
            margin-top: 4px;
        }}
        
        .time-badge {{
            font-size: 0.75em;
            color: #666;
            margin-bottom: 4px;
        }}
        
        .progress-info {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #888;
            margin-top: 8px;
        }}
        
        #confettiCanvas {{
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            pointer-events: none;
            z-index: 9999;
        }}
        
        /* ─── MOBILE: prevent double-tap zoom on interactive elements ─── */
        .play-pause-btn, .skip-btn, .lyric-segment, #focusModeBtn {{
            touch-action: manipulation;
        }}
        
        /* ─── MOBILE RESPONSIVE ─── */
        @media (max-width: 768px) {{
            /* Audio controls */
            .audio-controls {{
                gap: 8px;
                padding: 6px 4px;
                flex-wrap: wrap;
            }}
            .play-pause-btn {{
                width: 44px;
                height: 44px;
                font-size: 18px;
            }}
            .player-badges {{
                order: -1;
                width: 100%;
                justify-content: center;
                margin-left: 0;
                margin-bottom: 4px;
            }}
            .player-badge {{
                font-size: 0.7em;
                padding: 3px 8px;
            }}
            
            /* Seek bar — larger thumb for touch */
            .seek-row {{
                gap: 6px;
            }}
            .seek-row input[type="range"] {{
                height: 8px;
            }}
            .seek-row input[type="range"]::-webkit-slider-thumb {{
                width: 22px;
                height: 22px;
            }}
            .seek-row input[type="range"]::-moz-range-thumb {{
                width: 22px;
                height: 22px;
            }}
            .skip-btn {{
                font-size: 0.7em;
                padding: 6px 10px;
                min-height: 32px;
            }}
            
            /* Progress info */
            .progress-info {{
                font-size: 0.75em;
                gap: 4px;
                flex-wrap: wrap;
            }}
            
            /* Song summary */
            .song-summary {{
                font-size: 0.8em;
                padding: 8px 10px;
                top: 90px;
            }}
            .song-summary-label {{
                font-size: 0.65em;
            }}
            
            /* Lyrics */
            .lyrics-container {{
                padding: 12px;
            }}
            .lyric-segment {{
                padding: 12px 10px;
                margin: 6px 0;
            }}
            .lyric-segment.active {{
                transform: scale(1);
            }}
            .original {{
                font-size: 1.1em;
            }}
            .romanized {{
                font-size: 0.95em;
            }}
            .translation {{
                font-size: 0.95em;
            }}
            .meaning {{
                font-size: 0.8em;
                padding: 6px 10px;
            }}
            .time-badge {{
                font-size: 0.65em;
            }}
            
            /* Focus mode */
            .focus-overlay {{
                padding: 20px;
            }}
            .focus-overlay .focus-original {{
                font-size: 1.4em;
            }}
            .focus-overlay .focus-romanized {{
                font-size: 1.1em;
                margin-bottom: 10px;
            }}
            .focus-overlay .focus-translation {{
                font-size: 1.2em;
            }}
            .focus-overlay .focus-meaning {{
                font-size: 0.85em;
            }}
            .focus-overlay .focus-hint {{
                font-size: 0.7em;
                bottom: 10px;
            }}
            
            /* Wave bars — fewer visible on mobile */
            .wave-bars {{
                gap: 4px;
                padding: 0 8px;
            }}
            .wave-bars span {{
                max-width: 10px;
            }}
        }}
        
        /* Small phones (≤400px) */
        @media (max-width: 400px) {{
            .karaoke-container {{
                padding: 8px;
            }}
            .play-pause-btn {{
                width: 40px;
                height: 40px;
                font-size: 16px;
            }}
            .original {{
                font-size: 1em;
            }}
            .translation {{
                font-size: 0.85em;
            }}
            .focus-overlay .focus-original {{
                font-size: 1.2em;
            }}
            .focus-overlay .focus-translation {{
                font-size: 1em;
            }}
        }}
    </style>
    
    <div id="karaokeContainer" class="karaoke-container" style="position: relative;" tabindex="0">
        <canvas id="confettiCanvas"></canvas>
        <div class="wave-bars"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
        <div class="lyrics-wrapper">
            <div class="audio-controls">
                <button type="button" class="play-pause-btn" id="playPauseBtn" title="Play / Pause" aria-label="Play or pause">▶</button>
                <div class="audio-wrap">
                <audio id="audioPlayer">
                    <source src="data:audio/{audio_format};base64,{audio_base64}" type="audio/{audio_format}">
                </audio>
                <div class="seek-row">
                    <button type="button" class="skip-btn" id="skipBackBtn" title="Back 10 seconds">−10s</button>
                    <input type="range" id="seekBar" min="0" max="100" value="0" step="0.1" title="Seek">
                    <button type="button" class="skip-btn" id="skipAheadBtn" title="Ahead 10 seconds">+10s</button>
                </div>
                <div class="progress-info">
                    <span id="currentSegment">Press play to start</span>
                    <span id="timeDisplay">0:00 / 0:00</span>
                    <button type="button" id="focusModeBtn" style="font-size:0.8em;padding:2px 8px;border-radius:6px;border:1px solid rgba(255,255,255,0.3);background:rgba(255,255,255,0.1);color:#ccc;cursor:pointer;">Focus mode</button>
                </div>
                </div>
                <div class="player-badges">
                    <span class="player-badge" id="langBadge">{lang_badge}</span>
                    <span class="player-badge" id="moodBadge">{mood_badge}</span>
                </div>
            </div>
            {f'<div class="song-summary"><div class="song-summary-label">Summary</div>{summary_escaped}</div>' if summary_escaped else ''}
            <div class="lyrics-container" id="lyricsContainer">
            </div>
            <div class="focus-overlay" id="focusOverlay">
                <div class="focus-content">
                    <div class="focus-original" id="focusOriginal">—</div>
                    <div class="focus-romanized" id="focusRomanized"></div>
                    <div class="focus-translation" id="focusTranslation">—</div>
                    <div class="focus-meaning" id="focusMeaning"></div>
                </div>
                <div class="focus-hint">Press F or click "Exit focus" above to close</div>
            </div>
        </div>
    </div>
    
    <script>
        (function() {{
            // ═══════════════════════════════════════════════════════════════════
            // KARAOKE SYNC ENGINE - First Principles Implementation
            // ═══════════════════════════════════════════════════════════════════
            // 
            // Architecture:
            // 1. All sync logic runs client-side (no server round-trips)
            // 2. DOM references cached once at startup
            // 3. Only update DOM when active line CHANGES (not every frame)
            // 4. Binary search for O(log n) line lookup
            // 5. Simple scroll model: auto-scroll resumes on line change
            //
            // ═══════════════════════════════════════════════════════════════════
            
            const segments = {segments_json};
            const audio = document.getElementById('audioPlayer');
            const container = document.getElementById('lyricsContainer');
            const currentSegmentDisplay = document.getElementById('currentSegment');
            const timeDisplay = document.getElementById('timeDisplay');
            const karaokeContainer = document.getElementById('karaokeContainer');
            const focusModeBtn = document.getElementById('focusModeBtn');
            const playPauseBtn = document.getElementById('playPauseBtn');
            const focusOverlay = document.getElementById('focusOverlay');
            const focusOriginal = document.getElementById('focusOriginal');
            const focusRomanized = document.getElementById('focusRomanized');
            const focusTranslation = document.getElementById('focusTranslation');
            const focusMeaning = document.getElementById('focusMeaning');
            
            // ─────────────────────────────────────────────────────────────────
            // STATE
            // ─────────────────────────────────────────────────────────────────
            let currentLineIndex = -1;           // Currently highlighted line
            let userScrolledAway = false;        // User manually scrolled
            let focusMode = false;               // Focus mode (big text overlay)
            let segmentElements = [];            // Cached DOM references
            
            // ─────────────────────────────────────────────────────────────────
            // AUDIO VISUALIZER - Real-time frequency analysis
            // ─────────────────────────────────────────────────────────────────
            // Uses Web Audio API to analyze frequencies and drive wave bars
            
            let audioContext = null;
            let analyser = null;
            let dataArray = null;
            let visualizerAnimationId = null;
            const waveBars = document.querySelectorAll('.wave-bars span');
            const NUM_BARS = waveBars.length;
            
            function initAudioVisualizer() {{
                if (audioContext) return;  // Already initialized
                
                try {{
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    analyser = audioContext.createAnalyser();
                    
                    // Connect audio element to analyser
                    const source = audioContext.createMediaElementSource(audio);
                    source.connect(analyser);
                    analyser.connect(audioContext.destination);
                    
                    // Configure analyser for smooth visualization
                    analyser.fftSize = 256;  // 128 frequency bins
                    analyser.smoothingTimeConstant = 0.7;  // Smooth transitions
                    
                    const bufferLength = analyser.frequencyBinCount;  // 128
                    dataArray = new Uint8Array(bufferLength);
                    
                }} catch (e) {{
                    console.warn('Web Audio API not supported, using fallback animation');
                    useFallbackAnimation();
                }}
            }}
            
            function updateVisualizer() {{
                if (!analyser || !dataArray) return;
                
                analyser.getByteFrequencyData(dataArray);
                
                // Use only the lower half of the spectrum (bins 0–63): most musical energy
                // is there; high bins are often near zero so half the bars never moved.
                const useBins = Math.min(64, dataArray.length);
                const binsPerBar = useBins / NUM_BARS;
                
                waveBars.forEach((bar, i) => {{
                    const startBin = Math.floor(i * binsPerBar);
                    const endBin = Math.min(Math.floor((i + 1) * binsPerBar), useBins);
                    let value = 0;
                    for (let b = startBin; b < endBin; b++) {{
                        if (dataArray[b] > value) value = dataArray[b];
                    }}
                    const scale = 0.1 + (value / 255) * 0.9;
                    bar.style.transform = `scaleY(${{scale}})`;
                    bar.style.opacity = 0.4 + (value / 255) * 0.6;
                }});
            }}
            
            function startVisualizerLoop() {{
                if (visualizerAnimationId) return;
                
                function loop() {{
                    if (!audio.paused) {{
                        updateVisualizer();
                    }}
                    visualizerAnimationId = requestAnimationFrame(loop);
                }}
                loop();
            }}
            
            function stopVisualizerLoop() {{
                if (visualizerAnimationId) {{
                    cancelAnimationFrame(visualizerAnimationId);
                    visualizerAnimationId = null;
                }}
            }}
            
            // Fallback: subtle CSS animation if Web Audio fails
            function useFallbackAnimation() {{
                waveBars.forEach((bar, i) => {{
                    bar.style.animation = `waveFallback 1.5s ease-in-out infinite`;
                    bar.style.animationDelay = `${{i * 0.08}}s`;
                }});
                // Add fallback keyframes if not present
                if (!document.getElementById('fallbackWaveStyle')) {{
                    const style = document.createElement('style');
                    style.id = 'fallbackWaveStyle';
                    style.textContent = `
                        @keyframes waveFallback {{
                            0%, 100% {{ transform: scaleY(0.15); opacity: 0.5; }}
                            50% {{ transform: scaleY(0.6); opacity: 0.8; }}
                        }}
                    `;
                    document.head.appendChild(style);
                }}
            }}
            
            // Reset bars when paused
            function resetBarsToIdle() {{
                waveBars.forEach(bar => {{
                    bar.style.transform = 'scaleY(0.15)';
                    bar.style.opacity = '0.5';
                }});
            }}
            
            // Initialize visualizer on first play (audio context requires user gesture)
            audio.addEventListener('play', () => {{
                initAudioVisualizer();
                if (audioContext && audioContext.state === 'suspended') {{
                    audioContext.resume();
                }}
                startVisualizerLoop();
            }});
            
            audio.addEventListener('pause', () => {{
                // Don't stop the loop, but bars will naturally settle
                // since updateVisualizer checks audio.paused
                setTimeout(resetBarsToIdle, 100);
            }});
            
            audio.addEventListener('ended', () => {{
                resetBarsToIdle();
            }});
            
            // ─────────────────────────────────────────────────────────────────
            // INITIALIZATION - Build DOM and cache references
            // ─────────────────────────────────────────────────────────────────
            function formatTime(seconds) {{
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
            }}
            
            // Build lyrics HTML once
            let lyricsHTML = '';
            segments.forEach((seg, idx) => {{
                const text = seg.text || '';
                const romanized = seg.romanized || '';
                const translation = seg.translation || '';
                const meaning = seg.meaning || '';
                lyricsHTML += `
                    <div class="lyric-segment" id="segment-${{idx}}" data-idx="${{idx}}" data-start="${{seg.start}}" data-end="${{seg.end}}">
                        <div class="time-badge">${{formatTime(seg.start)}}</div>
                        <div class="original">${{text || '—'}}</div>
                        ${{romanized ? `<div class="romanized">${{romanized}}</div>` : ''}}
                        <div class="translation">${{translation || '(translating...)'}}</div>
                        ${{meaning ? `<div class="meaning">${{meaning}}</div>` : ''}}
                    </div>
                `;
            }});
            container.innerHTML = lyricsHTML;
            
            // Cache all segment DOM elements (do this ONCE, not every frame)
            segmentElements = segments.map((_, i) => document.getElementById(`segment-${{i}}`));
            
            // ─────────────────────────────────────────────────────────────────
            // BINARY SEARCH - Find active line at given time O(log n)
            // ─────────────────────────────────────────────────────────────────
            function findActiveLineIndex(time) {{
                if (segments.length === 0) return -1;
                
                let left = 0;
                let right = segments.length - 1;
                
                while (left <= right) {{
                    const mid = Math.floor((left + right) / 2);
                    const seg = segments[mid];
                    
                    if (time >= seg.start && time < seg.end) {{
                        return mid;  // Found it
                    }} else if (time < seg.start) {{
                        right = mid - 1;
                    }} else {{
                        left = mid + 1;
                    }}
                }}
                
                return -1;  // No active line (gap between segments or before/after)
            }}
            
            // ─────────────────────────────────────────────────────────────────
            // SCROLL HANDLING - Simple model
            // ─────────────────────────────────────────────────────────────────
            // When user scrolls: pause auto-scroll
            // When line changes: resume auto-scroll (user probably wants to follow along)
            
            let isScrolling = false;
            let scrollTimeout = null;
            
            container.addEventListener('scroll', () => {{
                // Mark that user is scrolling
                isScrolling = true;
                userScrolledAway = true;
                
                // Clear previous timeout
                clearTimeout(scrollTimeout);
                
                // After scroll stops, mark scrolling as done
                scrollTimeout = setTimeout(() => {{
                    isScrolling = false;
                }}, 150);
            }}, {{ passive: true }});
            
            function scrollToLine(idx) {{
                if (idx < 0 || !segmentElements[idx]) return;
                if (isScrolling) return;  // Don't fight with user scroll
                
                segmentElements[idx].scrollIntoView({{
                    behavior: 'smooth',
                    block: 'center'
                }});
            }}
            
            // ─────────────────────────────────────────────────────────────────
            // LINE UPDATE - Only update DOM when line changes
            // ─────────────────────────────────────────────────────────────────
            function updateActiveLine(newIndex) {{
                // Skip if no change
                if (newIndex === currentLineIndex) return;
                
                const oldIndex = currentLineIndex;
                currentLineIndex = newIndex;
                
                // Remove 'active' from old line, add 'past'
                if (oldIndex >= 0 && segmentElements[oldIndex]) {{
                    segmentElements[oldIndex].classList.remove('active');
                    segmentElements[oldIndex].classList.add('past');
                }}
                
                // Add 'active' to new line
                if (newIndex >= 0 && segmentElements[newIndex]) {{
                    segmentElements[newIndex].classList.add('active');
                    segmentElements[newIndex].classList.remove('past');
                    
                    // Update segment counter
                    currentSegmentDisplay.textContent = `Line ${{newIndex + 1}} of ${{segments.length}}`;
                    
                    // Update focus mode if active
                    if (focusMode) {{
                        updateFocusContent(newIndex);
                    }}
                    
                    // Auto-scroll on line change (resets userScrolledAway)
                    // This is the key insight: resume auto-scroll when a NEW line starts
                    userScrolledAway = false;
                    scrollToLine(newIndex);
                }} else {{
                    currentSegmentDisplay.textContent = segments.length > 0 ? 'Press play to start' : 'No lyrics';
                }}
                
                // When we seek backwards, need to un-mark "past" lines
                if (newIndex >= 0 && oldIndex > newIndex) {{
                    for (let i = newIndex + 1; i <= oldIndex && i < segmentElements.length; i++) {{
                        if (segmentElements[i]) {{
                            segmentElements[i].classList.remove('past');
                        }}
                    }}
                }}
                
                // When we seek forwards, mark skipped lines as "past"
                if (newIndex > 0 && (oldIndex < 0 || newIndex > oldIndex + 1)) {{
                    for (let i = 0; i < newIndex; i++) {{
                        if (segmentElements[i] && !segmentElements[i].classList.contains('past')) {{
                            segmentElements[i].classList.add('past');
                        }}
                    }}
                }}
            }}
            
            // ─────────────────────────────────────────────────────────────────
            // MAIN SYNC LOOP - timeupdate event
            // ─────────────────────────────────────────────────────────────────
            audio.addEventListener('timeupdate', () => {{
                const currentTime = audio.currentTime;
                const duration = audio.duration || 0;
                
                // Update time display (this is cheap, do every frame)
                timeDisplay.textContent = `${{formatTime(currentTime)}} / ${{formatTime(duration)}}`;
                
                // Update seek bar (only when user is not dragging)
                if (seekBar && !userSeeking) {{
                    seekBar.max = duration || 100;
                    seekBar.value = currentTime;
                }}
                
                // Find and update active line (only updates DOM if changed)
                const newIndex = findActiveLineIndex(currentTime);
                updateActiveLine(newIndex);
                
                // Confetti near end (once)
                if (duration > 5 && currentTime >= duration - 5 && !window._celebrationShown) {{
                    window._celebrationShown = true;
                    triggerConfetti();
                }}
            }});
            
            // ─────────────────────────────────────────────────────────────────
            // FOCUS MODE
            // ─────────────────────────────────────────────────────────────────
            function updateFocusContent(idx) {{
                if (idx < 0 || idx >= segments.length) {{
                    focusOriginal.textContent = '—';
                    focusRomanized.textContent = '';
                    focusTranslation.textContent = '—';
                    focusMeaning.textContent = '';
                    return;
                }}
                const seg = segments[idx];
                focusOriginal.textContent = seg.text || '—';
                focusRomanized.textContent = seg.romanized || '';
                focusRomanized.style.display = seg.romanized ? 'block' : 'none';
                focusTranslation.textContent = seg.translation || '—';
                focusMeaning.textContent = seg.meaning || '';
                focusMeaning.style.display = seg.meaning ? 'block' : 'none';
            }}
            
            function setFocusMode(on) {{
                focusMode = on;
                focusOverlay.classList.toggle('show', focusMode);
                container.style.visibility = focusMode ? 'hidden' : 'visible';
                focusModeBtn.textContent = focusMode ? 'Exit focus' : 'Focus mode';
                if (focusMode && currentLineIndex >= 0) {{
                    updateFocusContent(currentLineIndex);
                }}
            }}
            
            // ─────────────────────────────────────────────────────────────────
            // PLAY/PAUSE BUTTON
            // ─────────────────────────────────────────────────────────────────
            function updatePlayPauseIcon() {{
                if (playPauseBtn) playPauseBtn.textContent = audio.paused ? '▶' : '❚❚';
            }}
            
            if (playPauseBtn) {{
                playPauseBtn.addEventListener('click', (e) => {{
                    e.preventDefault();
                    if (audio.paused) audio.play(); else audio.pause();
                }});
            }}
            audio.addEventListener('play', updatePlayPauseIcon);
            audio.addEventListener('pause', updatePlayPauseIcon);
            updatePlayPauseIcon();
            
            // ─────────────────────────────────────────────────────────────────
            // SEEK BAR + SKIP ±10s
            // ─────────────────────────────────────────────────────────────────
            const seekBar = document.getElementById('seekBar');
            const skipBackBtn = document.getElementById('skipBackBtn');
            const skipAheadBtn = document.getElementById('skipAheadBtn');
            let userSeeking = false;
            
            audio.addEventListener('loadedmetadata', () => {{
                if (seekBar) seekBar.max = audio.duration || 0;
            }});
            audio.addEventListener('durationchange', () => {{
                if (seekBar) seekBar.max = audio.duration || 0;
            }});
            
            if (seekBar) {{
                seekBar.addEventListener('input', () => {{
                    userSeeking = true;
                    const t = parseFloat(seekBar.value);
                    if (!isNaN(t)) audio.currentTime = t;
                }});
                seekBar.addEventListener('change', () => {{ userSeeking = false; }});
            }}
            
            if (skipBackBtn) {{
                skipBackBtn.addEventListener('click', (e) => {{
                    e.preventDefault();
                    audio.currentTime = Math.max(0, audio.currentTime - 10);
                }});
            }}
            if (skipAheadBtn) {{
                skipAheadBtn.addEventListener('click', (e) => {{
                    e.preventDefault();
                    const d = audio.duration;
                    audio.currentTime = d ? Math.min(d, audio.currentTime + 10) : audio.currentTime + 10;
                }});
            }}
            
            // ─────────────────────────────────────────────────────────────────
            // CLICK TO SEEK
            // ─────────────────────────────────────────────────────────────────
            segmentElements.forEach((el, idx) => {{
                if (!el) return;
                el.addEventListener('click', (e) => {{
                    e.preventDefault();
                    const start = segments[idx].start;
                    audio.currentTime = start;
                    audio.play();
                }});
            }});
            
            // ─────────────────────────────────────────────────────────────────
            // KEYBOARD SHORTCUTS
            // ─────────────────────────────────────────────────────────────────
            karaokeContainer.addEventListener('mousedown', () => karaokeContainer.focus());
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'f' || e.key === 'F') {{
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                    e.preventDefault();
                    setFocusMode(!focusMode);
                }}
            }});
            
            if (focusModeBtn) {{
                focusModeBtn.addEventListener('click', (e) => {{
                    e.preventDefault();
                    setFocusMode(!focusMode);
                }});
            }}
            
            // ─────────────────────────────────────────────────────────────────
            // PAGE VISIBILITY - Pause when tab hidden
            // ─────────────────────────────────────────────────────────────────
            document.addEventListener('visibilitychange', () => {{
                if (document.hidden) audio.pause();
            }});
            window.addEventListener('pagehide', () => audio.pause());
            window.addEventListener('beforeunload', () => audio.pause());
            
            // ─────────────────────────────────────────────────────────────────
            // CONFETTI CELEBRATION
            // ─────────────────────────────────────────────────────────────────
            let confettiAnimationId = null;
            
            function triggerConfetti() {{
                const canvas = document.getElementById('confettiCanvas');
                if (!canvas) return;
                
                // Full viewport
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
                canvas.style.display = 'block';
                const ctx = canvas.getContext('2d');
                const colors = ['#00d4ff', '#ffd700', '#ff6b9d', '#7b68ee', '#50fa7b', '#ff79c6', '#ff6347', '#40e0d0'];
                const particles = [];
                
                // Spawn confetti pieces: rectangles and circles falling from the top
                function spawnBatch(count) {{
                    for (let i = 0; i < count; i++) {{
                        particles.push({{
                            x: Math.random() * canvas.width,
                            y: -10 - Math.random() * 40,
                            vx: (Math.random() - 0.5) * 3,
                            vy: 2 + Math.random() * 4,
                            color: colors[Math.floor(Math.random() * colors.length)],
                            size: 5 + Math.random() * 7,
                            rotation: Math.random() * 360,
                            rotationSpeed: (Math.random() - 0.5) * 8,
                            shape: Math.random() > 0.5 ? 'rect' : 'circle',
                            wobble: Math.random() * Math.PI * 2,
                            wobbleSpeed: 0.03 + Math.random() * 0.05,
                            opacity: 0.8 + Math.random() * 0.2,
                        }});
                    }}
                }}
                
                // Initial burst
                spawnBatch(80);
                
                let lastSpawn = 0;
                
                function render(t) {{
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Spawn new confetti every 300ms to keep it raining
                    if (t - lastSpawn > 300) {{
                        spawnBatch(8);
                        lastSpawn = t;
                    }}
                    
                    // Update and draw
                    for (let i = particles.length - 1; i >= 0; i--) {{
                        const p = particles[i];
                        p.wobble += p.wobbleSpeed;
                        p.x += p.vx + Math.sin(p.wobble) * 0.8;
                        p.y += p.vy;
                        p.rotation += p.rotationSpeed;
                        p.vy += 0.02;  // gentle gravity
                        
                        // Remove if off-screen
                        if (p.y > canvas.height + 20) {{
                            particles.splice(i, 1);
                            continue;
                        }}
                        
                        ctx.save();
                        ctx.globalAlpha = p.opacity;
                        ctx.translate(p.x, p.y);
                        ctx.rotate(p.rotation * Math.PI / 180);
                        ctx.fillStyle = p.color;
                        
                        if (p.shape === 'rect') {{
                            ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
                        }} else {{
                            ctx.beginPath();
                            ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
                            ctx.fill();
                        }}
                        ctx.restore();
                    }}
                    
                    confettiAnimationId = requestAnimationFrame(render);
                }}
                
                confettiAnimationId = requestAnimationFrame(render);
            }}
            
            function stopConfetti() {{
                if (confettiAnimationId) {{
                    cancelAnimationFrame(confettiAnimationId);
                    confettiAnimationId = null;
                }}
                const canvas = document.getElementById('confettiCanvas');
                if (canvas) {{
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    canvas.style.display = 'none';
                }}
                window._celebrationShown = false;
            }}
            
            // Stop confetti when song is paused or seeked away from the end
            audio.addEventListener('pause', stopConfetti);
            audio.addEventListener('seeked', () => {{
                const d = audio.duration || 0;
                if (d > 5 && audio.currentTime < d - 5) {{
                    stopConfetti();
                }}
            }});
            
            // Resize confetti canvas on orientation change / window resize
            window.addEventListener('resize', () => {{
                const canvas = document.getElementById('confettiCanvas');
                if (canvas && canvas.style.display !== 'none') {{
                    canvas.width = window.innerWidth;
                    canvas.height = window.innerHeight;
                }}
            }});
        }})();
    </script>
    """
    
    return html

# Curated songs for Browse by Language tab (optional video_id for thumbnail)
CURATED_SONGS = {
    "🇫🇷 French": [
        {"title": "La Vie en Rose", "artist": "Édith Piaf", "query": "La Vie en Rose Edith Piaf official", "video_id": "sGP3lwDqDtw"},
        {"title": "Alors on danse", "artist": "Stromae", "query": "Stromae Alors on danse official", "video_id": "VHoT4N43jK8"},
    ],
    "🇪🇸 Spanish": [
        {"title": "Despacito", "artist": "Luis Fonsi", "query": "Despacito Luis Fonsi official video", "video_id": "kJQP7kiw5Fk"},
        {"title": "La Bicicleta", "artist": "Shakira", "query": "Shakira La Bicicleta official", "video_id": "-UV0QGLmYys"},
    ],
    "🇰🇷 Korean": [
        {"title": "Gangnam Style", "artist": "PSY", "query": "PSY Gangnam Style official", "video_id": "9bZkp7q19f0"},
        {"title": "Dynamite", "artist": "BTS", "query": "BTS Dynamite official MV", "video_id": "gdZLi9oWNZg"},
    ],
    "🇯🇵 Japanese": [
        {"title": "Lemon", "artist": "Kenshi Yonezu", "query": "Kenshi Yonezu Lemon official", "video_id": "SX_ViT4Ra7k"},
        {"title": "First Love", "artist": "Hikaru Utada", "query": "Hikaru Utada First Love", "video_id": "gVfR6BsPBpE"},
    ],
    "🇮🇹 Italian": [
        {"title": "Nel blu dipinto di blu (Volare)", "artist": "Domenico Modugno", "query": "Volare Domenico Modugno official", "video_id": ""},
        {"title": "Con te partirò", "artist": "Andrea Bocelli", "query": "Con te partiro Andrea Bocelli official", "video_id": ""},
    ],
    "🇵🇹 Portuguese (Brazil)": [
        {"title": "Garota de Ipanema", "artist": "Antônio Carlos Jobim", "query": "Garota de Ipanema Jobim official", "video_id": ""},
        {"title": "Ai se eu te pego", "artist": "Michel Teló", "query": "Ai se eu te pego Michel Telo official", "video_id": ""},
    ],
    "🇩🇪 German": [
        {"title": "99 Luftballons", "artist": "Nena", "query": "99 Luftballons Nena official", "video_id": ""},
        {"title": "Du hast", "artist": "Rammstein", "query": "Rammstein Du hast official", "video_id": ""},
    ],
    "🇮🇳 Hindi": [
        {"title": "Tum Hi Ho", "artist": "Arijit Singh", "query": "Tum Hi Ho Arijit Singh official", "video_id": ""},
        {"title": "Kesariya", "artist": "Arijit Singh", "query": "Kesariya Brahmastra official", "video_id": ""},
    ],
    "🇸🇦 Arabic": [
        {"title": "Habibi", "artist": "Mohamed Ramadan", "query": "Habibi Mohamed Ramadan official", "video_id": ""},
        {"title": "Enta Eih", "artist": "Nancy Ajram", "query": "Nancy Ajram Enta Eih official", "video_id": ""},
    ],
    "🇷🇺 Russian": [
        {"title": "Million Roses", "artist": "Alla Pugacheva", "query": "Million Roses Alla Pugacheva", "video_id": ""},
        {"title": "Kalinka", "artist": "Traditional", "query": "Kalinka Russian folk official", "video_id": ""},
    ],
    "🇹🇷 Turkish": [
        {"title": "Gibi Gibi", "artist": "Sezen Aksu", "query": "Sezen Aksu Gibi Gibi official", "video_id": ""},
        {"title": "Düm Tek Tek", "artist": "Hadise", "query": "Dum Tek Tek Hadise Eurovision", "video_id": ""},
    ],
    "🇨🇳 Mandarin Chinese": [
        {"title": "Tian Mi Mi", "artist": "Teresa Teng", "query": "Tian Mi Mi Teresa Teng official", "video_id": ""},
        {"title": "Qing Hua Ci", "artist": "Jay Chou", "query": "Qing Hua Ci Jay Chou official", "video_id": ""},
    ],
    "🇭🇰 Cantonese": [
        {"title": "Beyond the Sea", "artist": "Beyond", "query": "Beyond Hai Kuo Tian Kong official", "video_id": ""},
        {"title": "Monica", "artist": "Leslie Cheung", "query": "Leslie Cheung Monica official", "video_id": ""},
    ],
    "🇹🇭 Thai": [
        {"title": "Phoenix", "artist": "Bambam", "query": "Bambam Phoenix official", "video_id": ""},
        {"title": "Dai Yin Mai", "artist": "Palmy", "query": "Palmy Dai Yin Mai official", "video_id": ""},
    ],
    "🇻🇳 Vietnamese": [
        {"title": "See Tinh", "artist": "Hoang Thuy Linh", "query": "See Tinh Hoang Thuy Linh official", "video_id": ""},
        {"title": "Ghen", "artist": "Min & Erik", "query": "Ghen Min Erik official", "video_id": ""},
    ],
    "🇮🇩 Indonesian": [
        {"title": "Kopi Dangdut", "artist": "Fahmi Shahab", "query": "Kopi Dangdut Fahmi Shahab official", "video_id": ""},
        {"title": "Lathi", "artist": "Weird Genius", "query": "Weird Genius Lathi official", "video_id": ""},
    ],
    "🇵🇭 Tagalog": [
        {"title": "Buwan", "artist": "Juan Karlos", "query": "Buwan Juan Karlos official", "video_id": ""},
        {"title": "Hanggang", "artist": "Wency Cornejo", "query": "Hanggang Wency Cornejo official", "video_id": ""},
    ],
    "🇸🇪 Swedish": [
        {"title": "Dancing Queen", "artist": "ABBA", "query": "ABBA Dancing Queen official", "video_id": ""},
        {"title": "Euphoria", "artist": "Loreen", "query": "Loreen Euphoria Eurovision official", "video_id": ""},
    ],
    "🇳🇱 Dutch": [
        {"title": "Venus", "artist": "Shocking Blue", "query": "Shocking Blue Venus official", "video_id": ""},
        {"title": "Zoutelande", "artist": "Bløf", "query": "Blof Zoutelande official", "video_id": ""},
    ],
    "🇵🇱 Polish": [
        {"title": "Dziwny jest ten świat", "artist": "Czesław Niemen", "query": "Dziwny jest ten swiat Niemen", "video_id": ""},
        {"title": "Przez twe oczy zielone", "artist": "Various", "query": "Przez twe oczy zielone official", "video_id": ""},
    ],
    "🇬🇷 Greek": [
        {"title": "Zorba's Dance", "artist": "Mikis Theodorakis", "query": "Zorba dance Greek official", "video_id": ""},
        {"title": "Mia Kardia", "artist": "Anna Vissi", "query": "Anna Vissi Mia Kardia official", "video_id": ""},
    ],
    "🇮🇱 Hebrew": [
        {"title": "Hallelujah", "artist": "Rita", "query": "Rita Hallelujah Hebrew official", "video_id": ""},
        {"title": "Diva", "artist": "Dana International", "query": "Dana International Diva Eurovision", "video_id": ""},
    ],
    "🇮🇷 Persian (Farsi)": [
        {"title": "Bebakhsh", "artist": "Googoosh", "query": "Googoosh Bebakhsh official", "video_id": ""},
        {"title": "Shabe Eshgh", "artist": "Ebi", "query": "Ebi Shabe Eshgh official", "video_id": ""},
    ],
    "🇧🇩 Bengali": [
        {"title": "Amar Shonar Bangla", "artist": "Rabindranath Tagore", "query": "Amar Shonar Bangla Bangladesh national", "video_id": ""},
        {"title": "Phire Esho", "artist": "Anupam Roy", "query": "Anupam Roy Phire Esho official", "video_id": ""},
    ],
    "🇮🇳 Tamil": [
        {"title": "Kolaveri Di", "artist": "Dhanush", "query": "Kolaveri Di Dhanush official", "video_id": ""},
        {"title": "Jai Ho", "artist": "A.R. Rahman", "query": "Jai Ho Slumdog Millionaire official", "video_id": ""},
    ],
    "🇮🇳 Telugu": [
        {"title": "Naatu Naatu", "artist": "Rahul Sipligunj", "query": "Naatu Naatu RRR official", "video_id": ""},
        {"title": "Bahubali", "artist": "M.M. Keeravani", "query": "Bahubali theme song official", "video_id": ""},
    ],
    "🇮🇳 Punjabi": [
        {"title": "Lemonade", "artist": "Diljit Dosanjh", "query": "Lemonade Diljit Dosanjh official", "video_id": ""},
        {"title": "High Rated Gabru", "artist": "Guru Randhawa", "query": "High Rated Gabru Guru Randhawa official", "video_id": ""},
    ],
    "🇺🇦 Ukrainian": [
        {"title": "Stefania", "artist": "Kalush Orchestra", "query": "Stefania Kalush Eurovision official", "video_id": ""},
        {"title": "Chervona Ruta", "artist": "Sofia Rotaru", "query": "Chervona Ruta Sofia Rotaru", "video_id": ""},
    ],
    "🇷🇴 Romanian": [
        {"title": "Dragostea din tei", "artist": "O-Zone", "query": "Dragostea din tei O-Zone official", "video_id": ""},
        {"title": "Stereo Love", "artist": "Edward Maya", "query": "Edward Maya Stereo Love official", "video_id": ""},
    ],
    "🇭🇺 Hungarian": [
        {"title": "Gloomy Sunday", "artist": "Rezső Seress", "query": "Gloomy Sunday Hungarian official", "video_id": ""},
        {"title": "Kinek mondjam el", "artist": "Viktor Király", "query": "Kinek mondjam el Viktor Kiraly", "video_id": ""},
    ],
    "🇨🇿 Czech": [
        {"title": "Holky z města", "artist": "Olympic", "query": "Olympic Holky z mesta", "video_id": ""},
        {"title": "Láska", "artist": "Lucie", "query": "Lucie Laska official", "video_id": ""},
    ],
    "🇳🇴 Norwegian": [
        {"title": "Fairytale", "artist": "Alexander Rybak", "query": "Alexander Rybak Fairytale Eurovision", "video_id": ""},
        {"title": "Take On Me", "artist": "a-ha", "query": "a-ha Take On Me official", "video_id": ""},
    ],
    "🇩🇰 Danish": [
        {"title": "Only Teardrops", "artist": "Emmelie de Forest", "query": "Only Teardrops Eurovision Denmark", "video_id": ""},
        {"title": "Smuk som et stjerneskud", "artist": "Medina", "query": "Medina Smuk som et stjerneskud", "video_id": ""},
    ],
    "🇫🇮 Finnish": [
        {"title": "Hard Rock Hallelujah", "artist": "Lordi", "query": "Lordi Hard Rock Hallelujah Eurovision", "video_id": ""},
        {"title": "Sandstorm", "artist": "Darude", "query": "Darude Sandstorm official", "video_id": ""},
    ],
    "🇲🇾 Malay": [
        {"title": "Bila Tiba Masanya", "artist": "Siti Nurhaliza", "query": "Siti Nurhaliza Bila Tiba Masanya", "video_id": ""},
        {"title": "Lelaki Teragung", "artist": "Dayang Nurfaizah", "query": "Dayang Nurfaizah Lelaki Teragung", "video_id": ""},
    ],
    "🇰🇪 Swahili": [
        {"title": "Jambo Bwana", "artist": "Them Mushrooms", "query": "Jambo Bwana Them Mushrooms", "video_id": ""},
        {"title": "Malaika", "artist": "Fadhili William", "query": "Malaika Fadhili William", "video_id": ""},
    ],
    "🇿🇦 Afrikaans": [
        {"title": "De la Rey", "artist": "Bok van Blerk", "query": "De la Rey Bok van Blerk", "video_id": ""},
        {"title": "Suzanne", "artist": "Leon Schuster", "query": "Leon Schuster Suzanne", "video_id": ""},
    ],
    "🇮🇪 Irish": [
        {"title": "The Foggy Dew", "artist": "The Chieftains", "query": "The Foggy Dew Chieftains official", "video_id": ""},
        {"title": "Oró Sé do Bheatha 'Bhaile", "artist": "Traditional", "query": "Oro Se do Bheatha Bhaile Irish", "video_id": ""},
    ],
    "🇬🇧 Welsh": [
        {"title": "Calon Lân", "artist": "Traditional", "query": "Calon Lan Welsh hymn", "video_id": ""},
        {"title": "Yma o Hyd", "artist": "Dafydd Iwan", "query": "Yma o Hyd Dafydd Iwan", "video_id": ""},
    ],
    "🇪🇸 Catalan": [
        {"title": "Ai coração", "artist": "María del Mar Bonet", "query": "Maria del Mar Bonet Ai coracao", "video_id": ""},
        {"title": "El cant dels ocells", "artist": "Pau Casals", "query": "El cant dels ocells Pau Casals", "video_id": ""},
    ],
    "🇵🇹 Portuguese (Portugal)": [
        {"title": "Fado Português", "artist": "Amália Rodrigues", "query": "Amalia Rodrigues Fado official", "video_id": ""},
        {"title": "Amar pelos dois", "artist": "Salvador Sobral", "query": "Amar pelos dois Eurovision Portugal", "video_id": ""},
    ],
    "🇬🇪 Georgian": [
        {"title": "Suliko", "artist": "Traditional", "query": "Suliko Georgian folk", "video_id": ""},
        {"title": "For You", "artist": "Nika Kocharov", "query": "Nika Kocharov Young Georgian Lolitaz Eurovision", "video_id": ""},
    ],
    "🇦🇲 Armenian": [
        {"title": "Qele Qele", "artist": "Sirusho", "query": "Sirusho Qele Qele Eurovision", "video_id": ""},
        {"title": "Jan Jan", "artist": "Sona", "query": "Sona Jan Jan Armenia", "video_id": ""},
    ],
    "🇦🇿 Azerbaijani": [
        {"title": "Always", "artist": "Aysel & Arash", "query": "Always Aysel Arash Eurovision", "video_id": ""},
        {"title": "Skeletons", "artist": "Dihaj", "query": "Dihaj Skeletons Eurovision", "video_id": ""},
    ],
    "🇪🇬 Egyptian Arabic": [
        {"title": "El Bint el Shalabeya", "artist": "Mohamed Mounir", "query": "Mohamed Mounir El Bint el Shalabeya", "video_id": ""},
        {"title": "Habibi ya nour el ain", "artist": "Amr Diab", "query": "Amr Diab Habibi ya nour el ain", "video_id": ""},
    ],
    "🇲🇽 Spanish (Mexico)": [
        {"title": "Cielito Lindo", "artist": "Traditional", "query": "Cielito Lindo Mexican official", "video_id": ""},
        {"title": "Amor Eterno", "artist": "Rocío Dúrcal", "query": "Rocio Durcal Amor Eterno official", "video_id": ""},
    ],
    "🇦🇷 Spanish (Argentina)": [
        {"title": "Cambalache", "artist": "Enrique Santos Discépolo", "query": "Cambalache tango official", "video_id": ""},
        {"title": "Bailando", "artist": "Paradisio", "query": "Paradisio Bailando official", "video_id": ""},
    ],
}

def _normalize_for_match(s: str) -> str:
    """Lowercase and strip for artist/language/mood comparison."""
    return (s or "").strip().lower()


def _artist_match(channel: str, artist: str) -> bool:
    """True if channel and artist are likely the same (one contains the other or close)."""
    c, a = _normalize_for_match(channel), _normalize_for_match(artist)
    if not c or not a:
        return False
    return a in c or c in a or a[:20] in c or c[:20] in a


def _song_key(title: str, subtitle: str) -> str:
    """Unique key for deduplication (title + artist/channel)."""
    return f"{_normalize_for_match(title)}|{_normalize_for_match(subtitle)}"


def get_suggested_songs(
    language: str,
    current_url: str,
    current_title: str,
    max_suggestions: int = 5,
    mood: str = None,
    channel: str = None,
) -> list:
    """
    Suggest songs for 'You might also like'.
    Priority: relevance (artist > mood > language), then history over curated.
    Dedupes by song identity; never suggests current song.
    """
    lang_lower = _normalize_for_match(language)
    mood_lower = _normalize_for_match(mood)
    seen_urls = {current_url}
    seen_keys = {_song_key(current_title, channel or "")}  # don't suggest current
    candidates = []

    # ─── 1. History: user's cached songs, scored by relevance (≤10 min) ───
    try:
        for song in get_cached_songs():
            url = song.get("url")
            title = song.get("title", "Unknown")
            if not url or url == current_url or title == current_title:
                continue
            if url in seen_urls:
                continue
            duration_sec = _parse_duration_to_seconds(song.get("duration"))
            if duration_sec is not None and duration_sec > MAX_SUGGESTION_DURATION_SEC:
                continue
            seen_urls.add(url)

            song_lang = _normalize_for_match(str(song.get("language") or ""))
            song_mood = _normalize_for_match(str(song.get("mood") or ""))
            song_channel = song.get("channel", "")

            score = 0
            reasons = []
            if channel and _artist_match(channel, song_channel):
                score += 3
                reasons.append("Same artist")
            if mood_lower and song_mood == mood_lower:
                score += 2
                reasons.append("Same mood")
            if lang_lower and song_lang == lang_lower:
                score += 1
                reasons.append("Same language")

            reason = " · ".join(reasons) if reasons else "From your history"
            candidates.append({
                "title": title,
                "subtitle": song_channel,
                "url": url,
                "cache_key": song.get("cache_key"),
                "thumbnail": song.get("thumbnail", ""),
                "type": "history",
                "reason": reason,
                "_score": score,
                "_key": _song_key(title, song_channel),
            })
    except Exception:
        pass

    # ─── 2. Curated: all languages that match (e.g. Spanish + Spanish (Mexico)) ───
    for key, songs in CURATED_SONGS.items():
        key_lower = key.lower()
        if lang_lower and lang_lower not in key_lower:
            continue
        for s in songs:
            title = s.get("title", "")
            artist = s.get("artist", "")
            key_id = _song_key(title, artist)
            if key_id in seen_keys:
                continue
            # Don't add curated if we already have this song from history
            if any(c.get("_key") == key_id for c in candidates):
                continue

            score = 2 if lang_lower else 0
            reason = "Same language" if lang_lower else "Curated pick"
            candidates.append({
                "title": title,
                "subtitle": artist,
                "query": s.get("query", ""),
                "video_id": s.get("video_id", ""),
                "type": "curated",
                "reason": reason,
                "_score": score,
                "_key": key_id,
            })
    # No break: we now add from every matching language key (e.g. all Spanish variants)

    # ─── 3. Sort: by score (desc), then history before curated ───
    def sort_key(c):
        s = c.get("_score", 0)
        is_history = 1 if c.get("type") == "history" else 0
        return (-s, -is_history)  # history first when score ties

    candidates.sort(key=sort_key)

    # ─── 4. Take up to max_suggestions, dedupe by _key ───
    out = []
    for c in candidates:
        if len(out) >= max_suggestions:
            break
        key_id = c.get("_key")
        if key_id in seen_keys:
            continue
        seen_keys.add(key_id)
        item = {k: v for k, v in c.items() if not k.startswith("_")}
        out.append(item)
    return out

# Page config
st.set_page_config(
    page_title="Surasa",
    page_icon="🎶",
    layout="wide"
)

# Mobile-friendly global styles
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    /* Streamlit container padding on mobile */
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        h1 { font-size: 1.5rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem; padding: 0.4rem 0.6rem; }
        /* Stack columns vertically on mobile */
        [data-testid="column"] { width: 100% !important; flex: 100% !important; min-width: 100% !important; }
        /* Larger tap targets for buttons */
        .stButton > button { min-height: 44px; font-size: 0.9rem; }
        /* Search input */
        .stTextInput input { font-size: 16px !important; }  /* prevents iOS zoom on focus */
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎶 Surasa")
st.caption("सुर + रस — Understand any song. Transcribe, translate, and feel the meaning.")

# Resolve History click: load from cache and rerun so song shows above tabs
pending_url = st.session_state.pop('pending_history_url', None)
pending_title = st.session_state.pop('pending_history_title', None)
pending_key = st.session_state.pop('pending_history_cache_key', None)
if pending_url is not None and pending_title is not None:
    cached = get_cached_result_by_key(pending_key) if pending_key else get_cached_result(pending_url, "auto")
    if cached:
        st.session_state['selected_url'] = pending_url
        st.session_state['selected_title'] = pending_title
        cached = dict(cached)
        cached['segments'] = merge_instrumental_segments(cached.get('segments', []))
        if not cached.get('suggested_songs'):
            meta = cached.get('_meta', {})
            lang = cached.get('language') or meta.get('language')
            mood = cached.get('mood') or meta.get('mood')
            channel = meta.get('channel')
            if lang or mood or channel:
                cached['suggested_songs'] = get_suggested_songs(
                    lang or "", meta.get('url', ''), meta.get('title', ''),
                    mood=mood, channel=channel
                )
        st.session_state['karaoke_data'] = cached
    else:
        st.session_state['selected_url'] = pending_url
        st.session_state['selected_title'] = pending_title
    st.rerun()

# Check if we have karaoke data to display (song is ready)
has_karaoke = 'karaoke_data' in st.session_state

# Processing bar (when a song is selected but not yet loaded)
processing_container = st.container()
if 'selected_url' in st.session_state and 'karaoke_data' not in st.session_state:
    with processing_container:
        title = st.session_state.get('selected_title', 'Song')
        st.info(f"**Preparing “{title}”** — transcribing and interpreting lyrics. This may take a minute.")

# Song block ABOVE tabs so it's visible when History is long
if has_karaoke:
    st.markdown(f"### 🎤 {st.session_state.get('selected_title', 'Now Playing')}")
    st.caption("Tap a lyric line to jump to that moment · **F** = focus mode (bigger text, no scroll)")
    
    data = st.session_state['karaoke_data']
    karaoke_html = create_karaoke_player(
        data['audio_base64'],
        data['segments'],
        data.get('audio_format', 'mpeg'),
        language=data.get('language'),
        mood=data.get('mood'),
        summary=data.get('summary'),
    )
    
    st.components.v1.html(karaoke_html, height=700, scrolling=False)
    
    # Suggested next (similar songs)
    suggested = data.get('suggested_songs') or []
    if suggested:
        st.markdown("**You might also like**")
        st.caption("Similar by artist, mood, or language — one click to play")
        for idx, s in enumerate(suggested):
            col_thumb, col_info, col_btn = st.columns([1, 4, 1])
            with col_thumb:
                if s.get('thumbnail'):
                    st.markdown(f'<img src="{html.escape(s["thumbnail"])}" width="80" style="border-radius: 8px;" />', unsafe_allow_html=True)
                elif s.get('video_id'):
                    st.markdown(f'<img src="https://img.youtube.com/vi/{html.escape(s["video_id"])}/mqdefault.jpg" width="80" style="border-radius: 8px;" />', unsafe_allow_html=True)
            with col_info:
                st.markdown(f"**{s.get('title', 'Unknown')}**")
                sub = s.get('subtitle', '')
                reason = s.get('reason', '')
                if reason:
                    st.caption(f"{sub}  ·  _{reason}_" if sub else f"_{reason}_")
                else:
                    st.caption(sub)
            with col_btn:
                if s.get('type') == 'history' and s.get('url') and s.get('cache_key'):
                    if st.button("▶ Play", key=f"sug_hist_{idx}_{s['cache_key']}", help="Play this song"):
                        st.session_state['pending_history_url'] = s['url']
                        st.session_state['pending_history_title'] = s['title']
                        st.session_state['pending_history_cache_key'] = s['cache_key']
                        st.rerun()
                elif s.get('type') == 'curated' and s.get('query'):
                    if st.button("▶ Play", key=f"sug_cur_{idx}", help="Play this song"):
                        with st.spinner("Finding video..."):
                            res = search_youtube(s['query'])
                        if res:
                            st.session_state.pop('karaoke_data', None)
                            st.session_state['selected_url'] = res[0]['url']
                            st.session_state['selected_title'] = res[0]['title']
                            st.session_state['auto_process'] = True
                            st.rerun()
        st.divider()
    
    # Download song + Download lyrics + Choose another song
    st.caption("Done with this one? Download the audio, lyrics, or pick another below.")
    col_dl, col_lyrics, col_choose = st.columns(3)
    title = st.session_state.get('selected_title', 'song')
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80].strip() or "song"
    with col_dl:
        ext = "mp3" if data.get("audio_format", "mpeg") == "mpeg" else "webm"
        file_name_audio = f"{safe_name}.{ext}"
        audio_bytes = base64.b64decode(data["audio_base64"])
        mime = "audio/mpeg" if ext == "mp3" else "audio/webm"
        st.download_button("⬇️ Download song", data=audio_bytes, file_name=file_name_audio, mime=mime, use_container_width=True)
    with col_lyrics:
        def _format_lyrics_time(sec):
            m = int(sec) // 60
            s = int(sec) % 60
            return f"{m}:{s:02d}"
        lines = [f"{data.get('language') or 'Lyrics'} · {data.get('mood') or ''}", f"{title}", ""]
        for seg in data.get("segments", []):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            lines.append(f"[{_format_lyrics_time(start)} - {_format_lyrics_time(end)}]")
            if seg.get("text"):
                lines.append(f"  {seg['text'].strip()}")
            if seg.get("romanized", "").strip():
                lines.append(f"  ({seg['romanized'].strip()})")
            if seg.get("translation", "").strip():
                lines.append(f"  → {seg['translation'].strip()}")
            if seg.get("meaning", "").strip():
                lines.append(f"  · {seg['meaning'].strip()}")
            lines.append("")
        lyrics_content = "\n".join(lines)
        lyrics_name = f"{safe_name}_lyrics.txt"
        st.download_button("📄 Download lyrics", data=lyrics_content.encode("utf-8"), file_name=lyrics_name, mime="text/plain; charset=utf-8", use_container_width=True)
    with col_choose:
        if st.button("🎶 Choose another song", use_container_width=True):
            for key in ['selected_url', 'selected_title', 'karaoke_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

st.divider()

# Tabs: always visible so user can search/browse/history while song plays or before
tab1, tab2, tab3 = st.tabs(["🔍 Search", "🌍 Browse", "📜 History"])

with tab1:
    search_query = st_searchbox(
        get_youtube_suggestions,
        key="song_search",
        placeholder="Search any song (e.g., La Vie en Rose, Despacito, Gangnam Style...)",
        clear_on_submit=False,
    )
    if search_query:
        with st.spinner("Finding songs..."):
            results = search_youtube(search_query)
        if results:
            st.markdown("### Pick one to start")
            for i, result in enumerate(results):
                thumb = result.get("thumbnail") or _youtube_thumbnail_url(_video_id_from_url(result["url"]))
                col_thumb, col_info, col_btn = st.columns([1, 4, 1])
                with col_thumb:
                    if thumb:
                        st.markdown(f'<img src="{html.escape(thumb)}" width="120" style="border-radius: 8px; max-width: 100%;" />', unsafe_allow_html=True)
                with col_info:
                    st.markdown(f"**{result['title']}**")
                    st.caption(f"{result['channel']} • {result['duration']}")
                with col_btn:
                    if st.button("▶ Play", key=f"select_{i}", type="primary", help="Play this song"):
                        st.session_state.pop('karaoke_data', None)
                        st.session_state['selected_url'] = result['url']
                        st.session_state['selected_title'] = result['title']
                        st.session_state['auto_process'] = True
                        st.rerun()
                st.divider()
    st.caption("Or paste a YouTube link")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url_input",
        label_visibility="collapsed"
    )
    if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
        st.session_state.pop('karaoke_data', None)
        st.session_state['selected_url'] = youtube_url
        st.session_state['selected_title'] = "YouTube Video"
        st.session_state['auto_process'] = True
        st.rerun()

with tab2:
    all_languages = list(CURATED_SONGS.keys())
    filter_options = ["All languages"] + all_languages
    selected = st.selectbox(
        "Language",
        options=filter_options,
        key="browse_language_filter"
    )
    if selected == "All languages":
        languages_to_show = all_languages
    else:
        languages_to_show = [selected]
    for lang in languages_to_show:
        if lang not in CURATED_SONGS:
            continue
        songs = CURATED_SONGS[lang]
        st.markdown(f"**Songs in {lang}**")
        for j, s in enumerate(songs):
            thumb_url = _youtube_thumbnail_url(s.get("video_id", "")) if s.get("video_id") else ""
            col_thumb, col_info, col_btn = st.columns([1, 4, 1])
            with col_thumb:
                if thumb_url:
                    st.markdown(f'<img src="{html.escape(thumb_url)}" width="120" style="border-radius: 8px; max-width: 100%; display: block;" />', unsafe_allow_html=True)
                else:
                    st.markdown("<div style='height:90px;'></div>", unsafe_allow_html=True)
            with col_info:
                st.markdown(f"{s['title']} — *{s['artist']}*")
            with col_btn:
                if st.button("▶ Play", key=f"browse_{lang}_{j}", type="primary", help="Play this song"):
                    with st.spinner("Finding video..."):
                        res = search_youtube(s['query'])
                    if res:
                        st.session_state.pop('karaoke_data', None)
                        st.session_state['selected_url'] = res[0]['url']
                        st.session_state['selected_title'] = res[0]['title']
                        st.session_state['auto_process'] = True
                        st.rerun()
                    else:
                        st.error("Could not find a video for this song.")
            st.divider()
            st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

with tab3:
    cached_songs = get_cached_songs()
    if st.session_state.get("confirm_clear_history"):
        st.warning("Clear all history? This will remove every song from this list. You can still search and play them again (they’ll be processed again).")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, clear all history", type="primary"):
                try:
                    for f in os.listdir(CACHE_DIR):
                        if f.endswith(".json"):
                            try:
                                os.remove(os.path.join(CACHE_DIR, f))
                            except Exception:
                                pass
                except Exception:
                    pass
                for key in ["confirm_clear_history", "karaoke_data", "selected_url", "selected_title"]:
                    st.session_state.pop(key, None)
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state.pop("confirm_clear_history", None)
                st.rerun()
    elif not cached_songs:
        st.caption("No songs yet. Play something from Search or Browse — it'll show up here for quick replay.")
    else:
        if st.button("🗑️ Clear history", help="Remove all songs from history"):
            st.session_state["confirm_clear_history"] = True
            st.rerun()
        st.caption("Replay any song you've already played.")
        for song in cached_songs:
            thumb_url = song.get("thumbnail") or _youtube_thumbnail_url(_video_id_from_url(song.get("url", "")))
            col_thumb, col_info, col_btn = st.columns([1, 4, 1])
            with col_thumb:
                if thumb_url:
                    st.markdown(f'<img src="{html.escape(thumb_url)}" width="120" style="border-radius: 8px; max-width: 100%;" />', unsafe_allow_html=True)
            with col_info:
                st.markdown(f"**{song['title']}**")
                channel = song.get('channel', '') or 'Unknown'
                duration = song.get('duration', '')
                if channel and channel != 'Unknown' and duration:
                    st.caption(f"{channel} • {duration}")
                elif channel and channel != 'Unknown':
                    st.caption(channel)
                elif duration:
                    st.caption(duration)
                elif song.get('cached_at'):
                    st.caption(f"Played on {song['cached_at']}")
            with col_btn:
                if st.button("▶ Play", key=f"hist_{song['cache_key']}", help="Play again"):
                    st.session_state['pending_history_url'] = song['url']
                    st.session_state['pending_history_title'] = song['title']
                    st.session_state['pending_history_cache_key'] = song['cache_key']
                    st.rerun()
            st.divider()

# Process selected song only when no song is playing
if not has_karaoke:
    if 'selected_url' in st.session_state:
        # Auto-process if no karaoke data yet
        should_process = 'karaoke_data' not in st.session_state
        
        if should_process:
            # Check cache first
            cached = get_cached_result(st.session_state['selected_url'], "auto")
            if cached:
                with processing_container:
                    st.success("⚡ Loaded from cache — ready to play!")
                cached = dict(cached)
                cached['segments'] = merge_instrumental_segments(cached.get('segments', []))
                if not cached.get('suggested_songs'):
                    meta = cached.get('_meta', {})
                    lang = cached.get('language') or meta.get('language')
                    mood = cached.get('mood') or meta.get('mood')
                    channel = meta.get('channel')
                    cached['suggested_songs'] = get_suggested_songs(
                        lang or "", meta.get('url', ''), meta.get('title', ''),
                        mood=mood, channel=channel
                    )
                st.session_state['karaoke_data'] = cached
                st.rerun()
            
            # Show processing status at the TOP (in the container we created earlier)
            with processing_container:
                # Create temp directory
                tmp_dir = tempfile.mkdtemp()
                
                try:
                    # Step indicators
                    steps = ["⬇️ Download", "🎤 Transcribe", "🔮 Interpret"]
                    
                    # Get video duration for time estimates
                    meta = _get_youtube_metadata(st.session_state['selected_url'])
                    duration_sec = meta.get('duration_seconds')
                    
                    def _est(step_name, dur_sec):
                        """Rough time estimates per step based on video length (seconds)."""
                        if dur_sec is None:
                            return "~1–2 min" if step_name == "Download" else "~1–3 min" if step_name == "Transcribe" else "~1–2 min"
                        mins = dur_sec / 60.0
                        if step_name == "Download":
                            return f"~{max(1, int(0.5 + 0.3 * mins))} min" if mins > 2 else "~30 sec"
                        if step_name == "Transcribe":
                            # Whisper ~0.5–1x realtime
                            return f"~{max(1, int(0.5 + mins * 0.8))}–{max(2, int(0.5 + mins * 1.2))} min"
                        if step_name == "Interpret":
                            return f"~{max(1, int(0.5 + mins * 0.3))}–{max(2, int(0.5 + mins * 0.5))} min"
                        return ""
                    
                    est_download = _est("Download", duration_sec)
                    est_transcribe = _est("Transcribe", duration_sec)
                    est_interpret = _est("Interpret", duration_sec)
                    
                    # Progress bar (only reaches 100% when fully done)
                    progress_bar = st.progress(0)
                    step_display = st.empty()
                    detail_display = st.empty()
                    time_display = st.empty()
                    
                    start_time = time_module.time()
                    
                    def update_progress(step_num, detail="", time_remaining=""):
                        """Update progress UI. Bar reaches 100% only when all steps are done."""
                        progress = (0.1, 0.4, 0.7)[step_num - 1]  # 10%, 40%, 70% per step start
                        progress_bar.progress(progress)
                        
                        step_text = "  →  ".join([
                            f"**{s}**" if i == step_num - 1 else f"~~{s}~~" if i < step_num - 1 else s
                            for i, s in enumerate(steps)
                        ])
                        step_display.markdown(f"Step {step_num}/3: {step_text}")
                        
                        if detail:
                            detail_display.caption(detail)
                        
                        elapsed = time_module.time() - start_time
                        tr = f" · Est. {time_remaining} left" if time_remaining else ""
                        time_display.caption(f"⏱️ {elapsed:.1f}s elapsed{tr}")
                    
                    # Step 1: Download
                    update_progress(1, "Fetching audio from YouTube...", est_download)
                    download_messages = [
                        "Connecting to YouTube...",
                        "Downloading audio stream...",
                        "Converting to MP3...",
                    ]
                    with animated_status(detail_display, download_messages):
                        audio_path = download_audio(st.session_state['selected_url'], tmp_dir)
                    
                    # Step 2: Transcribe (auto-detect language)
                    update_progress(2, "Using Whisper AI (auto-detecting language)...", est_transcribe)
                    transcribe_messages = [
                        "Uploading audio to OpenAI...",
                        "Whisper is analyzing the audio...",
                        "Auto-detecting language...",
                        "Identifying lyrics and timestamps...",
                        "This can take 30-60 seconds for longer songs...",
                        "Still processing... hang tight!",
                    ]
                    with animated_status(detail_display, transcribe_messages, interval=2.0):
                        segments = transcribe_with_timestamps(audio_path)
                    
                    # Count unique segments for optimization info
                    text_segments = [s for s in segments if s['text'].strip()]
                    unique_count = len(set(s['text'].strip().lower() for s in text_segments))
                    detail_display.caption(f"✓ Found {len(text_segments)} lyric lines ({unique_count} unique)")
                    time_module.sleep(0.5)  # Brief pause to show the count
                    
                    # Step 3: Interpret with Claude Sonnet
                    update_progress(3, f"Claude Sonnet interpreting {unique_count} unique lines...")
                    interpret_messages = [
                        "Sending lyrics to Claude Sonnet...",
                        "Generating phonetic pronunciations...",
                        "Crafting poetic translations...",
                        "Analyzing cultural context...",
                        "Finding metaphors and idioms...",
                        "Exploring emotional subtext...",
                        "Building rich interpretations...",
                    ]
                    with animated_status(detail_display, interpret_messages, interval=2.5):
                        interpreted_segments = interpret_segments(segments)
                        interpreted_segments = merge_instrumental_segments(interpreted_segments)
                    
                    # Language, mood, and summary for badges, theme, and card
                    lang, mood, summary = get_language_and_mood(interpreted_segments)
                    current_url = st.session_state['selected_url']
                    current_title = st.session_state.get('selected_title', 'Unknown')
                    channel = meta.get('channel')
                    suggested_songs = get_suggested_songs(
                        lang, current_url, current_title,
                        mood=mood, channel=channel
                    )
                    
                    # Final: Build player
                    detail_display.caption("Building karaoke player...")
                    audio_base64 = get_audio_base64(audio_path)
                    
                    # Determine audio format
                    audio_ext = os.path.splitext(audio_path)[1].lstrip('.')
                    if audio_ext == 'webm':
                        audio_format = 'webm'
                    else:
                        audio_format = 'mpeg'
                    
                    # Only now is the progress bar complete
                    progress_bar.progress(1.0)
                    total_time = time_module.time() - start_time
                    step_display.markdown("✅ **Ready to play!**")
                    detail_display.caption(f"Processed {len(text_segments)} lines in {total_time:.1f}s")
                    time_display.empty()
                    
                    # Store data for display (language, mood, summary, suggested_songs for badges and card)
                    karaoke_data = {
                        'audio_base64': audio_base64,
                        'segments': interpreted_segments,
                        'audio_format': audio_format,
                        'language': lang,
                        'mood': mood,
                        'summary': summary,
                        'suggested_songs': suggested_songs,
                    }
                    st.session_state['karaoke_data'] = karaoke_data
                    
                    # Save to cache for next time (include language/mood in data so _meta can store them)
                    save_to_cache(
                        current_url, "auto", karaoke_data,
                        title=current_title
                    )
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    if 'selected_url' not in st.session_state:
        with st.expander("How it works"):
            st.markdown("""
            1. **Search** or paste a link → we find the song.
            2. **AI transcribes** lyrics and detects the language.
            3. **AI interprets** meaning, idioms, and context.
            4. **Karaoke mode** syncs lyrics as you play — tap a line to jump, press **F** for focus mode.

            **Best for:** understanding songs in any language, learning idioms and culture, singing along with romanization.
            """)

st.divider()
st.caption("© 2026 Abhinav Deshmukh · Lyrics and interpretations are AI-generated; use for learning only.")