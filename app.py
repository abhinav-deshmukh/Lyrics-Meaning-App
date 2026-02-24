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
    """Save processed song to cache with metadata for history.
    Channel and duration are left for get_cached_songs() to backfill from YouTube when needed (avoids blocking on yt-dlp here).
    """
    import time
    data = dict(data)
    data['_meta'] = {
        'url': url,
        'title': title or 'Unknown',
        'cached_at': time.strftime('%Y-%m-%d %H:%M'),
        'thumbnail': _youtube_thumbnail_url(_video_id_from_url(url)),
        'channel': 'Unknown',
        'duration': '',
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

1. **original**: The exact text as given (copy it exactly).
2. **romanized**: Phonetic pronunciation in Latin script (if the original uses non-Latin script; otherwise leave empty).
3. **translation**: Natural English translation. Always output English. Never copy the original into this field — e.g. "Sabes que ya llevo un rato mirándote" must become "You know I've been watching you for a while", not the Spanish again. For fillers like "Oh" or "Hey" you may repeat the word.
4. **meaning**: 1-2 sentences explaining cultural context, idioms, wordplay, or emotional subtext (optional for fillers).

Output exactly one JSON object per line, in the same order as the input. No extra keys or commentary. Only a valid JSON array.
Format: [{{"original":"...","romanized":"...","translation":"...","meaning":"..."}}]
{language_line}
Lines to interpret:
{segments}
"""
# Optional line injected when language is known (e.g. "The lyrics are in: Spanish.")
INTERPRETATION_PROMPT_LANGUAGE_LINE = "The lyrics are in: {language}."
# When source is not English, insist on English-only translation (avoids model echoing original).
INTERPRETATION_PROMPT_TRANSLATE_LINE = "Translate every line to English in the 'translation' field; do not copy the original text into translation."

# Whisper returns ISO 639-1 codes (e.g. 'es', 'ko'). Map to readable names for the prompt.
_WHISPER_LANG_TO_DISPLAY = {
    "es": "Spanish", "en": "English", "ko": "Korean", "ja": "Japanese", "fr": "French",
    "de": "German", "pt": "Portuguese", "it": "Italian", "ru": "Russian", "hi": "Hindi",
    "ar": "Arabic", "zh": "Chinese", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
    "tr": "Turkish", "pl": "Polish", "nl": "Dutch", "sv": "Swedish", "el": "Greek",
    "he": "Hebrew", "fa": "Persian", "uk": "Ukrainian", "ro": "Romanian", "hu": "Hungarian",
}


def _format_interpretation_prompt(segments_text: str, language_hint: str = None, insist_english: bool = False) -> str:
    """Build interpretation prompt with optional language hint (Whisper code or display name)."""
    language_line = ""
    if language_hint:
        try:
            display = _WHISPER_LANG_TO_DISPLAY.get(language_hint.strip().lower(), language_hint)
            language_line = "\n" + INTERPRETATION_PROMPT_LANGUAGE_LINE.format(language=display) + "\n\n"
            # For non-English lyrics, insist on English-only translation to avoid model echoing original
            if insist_english or (language_hint.strip().lower() != "en"):
                language_line += INTERPRETATION_PROMPT_TRANSLATE_LINE + "\n\n"
        except Exception:
            pass
    return INTERPRETATION_PROMPT.format(segments=segments_text, language_line=language_line)

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

COBALT_API_URL = os.getenv("COBALT_API_URL", "https://cobalt-production-b880.up.railway.app")


def _download_via_cobalt(url: str, output_dir: str) -> Optional[str]:
    """
    Download audio via Cobalt API (fast, no JS runtime needed).
    Returns file path on success, None on failure.
    """
    import time as _time
    cobalt_url = COBALT_API_URL.rstrip("/")

    for attempt in range(2):
        try:
            payload = json.dumps({
                "url": url,
                "downloadMode": "audio",
                "audioFormat": "mp3",
                "audioBitrate": "128",
            }).encode("utf-8")

            req = urllib.request.Request(
                cobalt_url,
                data=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            status = data.get("status")
            download_url = data.get("url")

            if status in ("tunnel", "redirect") and download_url:
                # Stream the audio file to disk
                out_path = os.path.join(output_dir, "audio.mp3")
                dl_req = urllib.request.Request(download_url)
                with urllib.request.urlopen(dl_req, timeout=120) as stream:
                    with open(out_path, "wb") as f:
                        while True:
                            chunk = stream.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)

                if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                    return out_path
                # File too small — remove and retry
                try:
                    os.remove(out_path)
                except Exception:
                    pass

        except Exception:
            pass

        if attempt < 1:
            _time.sleep(1)

    return None


def _download_via_ytdlp(url: str, output_dir: str) -> Optional[str]:
    """
    Download audio via yt-dlp (fallback). Returns file path on success, None on failure.
    """
    import time as _time
    output_template = os.path.join(output_dir, "audio.%(ext)s")

    for attempt in range(2):
        # Clean up partial downloads from previous attempts
        if attempt > 0:
            for f in os.listdir(output_dir):
                if f.startswith("audio."):
                    try:
                        os.remove(os.path.join(output_dir, f))
                    except Exception:
                        pass

        try:
            result = subprocess.run(
                ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "9",
                 "-o", output_template, "--no-playlist", url],
                capture_output=True, text=True, timeout=120
            )

            for f in os.listdir(output_dir):
                if f.startswith("audio."):
                    filepath = os.path.join(output_dir, f)
                    if os.path.getsize(filepath) > 1024:
                        return filepath
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
                    break

        except Exception:
            pass

        if attempt < 1:
            _time.sleep(2)

    return None


def download_audio(url: str, output_dir: str) -> str:
    """
    Download audio from YouTube URL.
    Strategy: Cobalt API first (fast, no JS runtime needed), yt-dlp fallback.
    """
    # Try Cobalt first — much faster and doesn't need Node.js
    path = _download_via_cobalt(url, output_dir)
    if path:
        return path

    # Fallback to yt-dlp
    path = _download_via_ytdlp(url, output_dir)
    if path:
        return path

    raise Exception("Download failed: both Cobalt API and yt-dlp were unable to fetch the audio.")

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

# Phrases Whisper sometimes echoes as hallucination (e.g. when prompt was used). Replace with ♪.
_KNOWN_HALLUCINATION_PHRASES = (
    "Lyrics of a song. Transcribe the singing. May be in any language.",
    "Lyrics of a song. Transcribe the singing. May be any language.",
)


def _transcribe_one_file(audio_path: str, language: str, client) -> tuple:
    """Single-file transcription. Returns (segments, detected_language or None)."""
    with open(audio_path, "rb") as audio_file:
        params = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
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
    detected_lang = getattr(transcript, 'language', None) or None
    return (segments, detected_lang)


def _transcribe_chunk_with_retry(chunk_path: str, offset_sec: float, language: str, client, max_retries: int = 3) -> tuple:
    """Transcribe a single chunk with per-chunk retry. Returns (segments with offsets applied, detected_language or None)."""
    import time
    for attempt in range(max_retries):
        try:
            segs, detected_lang = _transcribe_one_file(chunk_path, language, client)
            out = [{'start': s['start'] + offset_sec, 'end': s['end'] + offset_sec, 'text': s['text']} for s in segs]
            return (out, detected_lang)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return ([], None)


def _is_hallucinated_segment(seg: dict) -> bool:
    """Check if a segment looks like a Whisper hallucination (known phrases, very short repeated, etc.)."""
    text = (seg.get("text") or "").strip()
    if not text:
        return True
    if text in _KNOWN_HALLUCINATION_PHRASES:
        return True
    if text == "♪":
        return True
    return False


def _chunk_looks_hallucinated(chunk_segments: list) -> bool:
    """True if a chunk's segments are mostly hallucinated or have repeated short text."""
    if not chunk_segments:
        return True
    hallucinated = sum(1 for s in chunk_segments if _is_hallucinated_segment(s))
    if hallucinated >= len(chunk_segments) * 0.5:
        return True
    # Check for repeated short text (sign of hallucination on silence/music)
    texts = [(s.get("text") or "").strip().lower() for s in chunk_segments if (s.get("text") or "").strip()]
    if texts:
        from collections import Counter
        counts = Counter(texts)
        most_common_text, most_common_count = counts.most_common(1)[0]
        if most_common_count >= 3 and len(most_common_text) < 60:
            return True
    return False


def transcribe_with_timestamps(audio_path: str, language: str = None) -> tuple:
    """
    Transcribe audio with timestamps (two-pass for quality).
    Pass 1: Transcribe all chunks in parallel without language hint.
    Pass 2: If we detected a language AND some chunks look hallucinated, re-transcribe
             those chunks with the detected language as a hint. This recovers lyrics
             that Whisper missed (e.g. devotional songs with heavy background music).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = OpenAI()
    chunks = _split_audio_into_chunks(audio_path)
    created_chunk_files = [p for p, _ in chunks if p != audio_path]

    def _cleanup():
        for p in created_chunk_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # ── Pass 1: transcribe all chunks in parallel (no language hint) ──
    chunk_results = {}  # {(path, offset): (segments, lang)}
    detected_language = None
    max_workers = min(len(chunks), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_transcribe_chunk_with_retry, path, off, language, client): (path, off)
            for path, off in chunks
        }
        for fut in as_completed(futures):
            key = futures[fut]
            segs, lang = fut.result()
            chunk_results[key] = (segs, lang)
            if lang and detected_language is None:
                detected_language = lang

    # ── Pass 2: re-transcribe hallucinated chunks with language hint ──
    if detected_language and not language:
        retranscribe_keys = []
        for key in chunks:
            segs, _ = chunk_results.get(key, ([], None))
            if _chunk_looks_hallucinated(segs):
                retranscribe_keys.append(key)

        if retranscribe_keys and len(retranscribe_keys) < len(chunks):
            with ThreadPoolExecutor(max_workers=min(len(retranscribe_keys), 4)) as executor:
                futures = {
                    executor.submit(_transcribe_chunk_with_retry, path, off, detected_language, client): (path, off)
                    for path, off in retranscribe_keys
                }
                for fut in as_completed(futures):
                    key = futures[fut]
                    new_segs, new_lang = fut.result()
                    old_segs, _ = chunk_results[key]
                    # Use new result if it has more real content
                    if new_segs and not _chunk_looks_hallucinated(new_segs):
                        chunk_results[key] = (new_segs, new_lang or detected_language)

    # ── Merge all chunk results ──
    all_segments = []
    for key in chunks:
        segs, _ = chunk_results.get(key, ([], None))
        if segs:
            all_segments.extend(segs)
    all_segments.sort(key=lambda x: x['start'])

    # Quality check: if everything is empty/short, one full retry
    text_segments = [s for s in all_segments if s['text'].strip()]
    total_text = ' '.join(s['text'] for s in text_segments).strip() if text_segments else ''
    if not text_segments or len(total_text) < 20:
        time.sleep(2)
        all_segments = []
        detected_language = None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_transcribe_chunk_with_retry, path, off, language, client): (path, off)
                for path, off in chunks
            }
            for fut in as_completed(futures):
                segs, lang = fut.result()
                if segs:
                    all_segments.extend(segs)
                if lang and detected_language is None:
                    detected_language = lang
        all_segments.sort(key=lambda x: x['start'])

    _cleanup()
    return (all_segments, detected_language)


INTERPRETATION_BATCH_SIZE = 25  # Lines per API call — prevents max_tokens truncation


def _extract_json_array(response_text: str) -> list:
    """Robustly extract a JSON array from an LLM response, handling markdown fences and extra text."""
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

    return json.loads(json_text)


def _interpret_batch(texts: list, client, language_hint: str = None, max_retries: int = 3) -> dict:
    """
    Interpret a batch of unique lyric lines. Returns dict {text_lower: interp_dict}.
    Retries on network errors and JSON parse failures. If the model echoes the original
    in the translation field (common for Spanish etc.), retries once with a stronger
    English-only instruction.
    """
    import time
    if not texts:
        return {}

    segments_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

    def _parse_batch_result(texts_list, interpretations, accept_echo_fillers: bool = True) -> dict:
        lookup = {}
        for i, text in enumerate(texts_list):
            if i < len(interpretations):
                interp = interpretations[i]
                trans = (interp.get('translation') or '').strip()
                orig = (interp.get('original') or text or '').strip()
                if trans and trans.lower() != orig.lower():
                    lookup[text.strip().lower()] = interp
                elif accept_echo_fillers and trans and len(orig.split()) <= 2 and trans.lower() == orig.lower():
                    lookup[text.strip().lower()] = interp
        return lookup

    def _echo_ratio(texts_list, interpretations) -> float:
        """Fraction of multi-word lines where translation == original (echo)."""
        multiword = 0
        echoes = 0
        for i, text in enumerate(texts_list):
            orig = (text or '').strip()
            if len(orig.split()) <= 2:
                continue
            multiword += 1
            if i < len(interpretations):
                trans = (interpretations[i].get('translation') or '').strip()
                if trans and trans.lower() == orig.lower():
                    echoes += 1
        return echoes / multiword if multiword else 0.0

    # First attempt
    prompt_content = _format_interpretation_prompt(segments_text, language_hint)
    last_lookup = None  # set when we retry due to high echo ratio
    last_echo_ratio = 1.0

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt_content}]
            )
            interpretations = _extract_json_array(response.content[0].text)
            lookup = _parse_batch_result(texts, interpretations)
            echo_ratio = _echo_ratio(texts, interpretations)

            # If most multi-word lines were echoed and we have a non-English hint, retry once with stronger prompt
            if echo_ratio > 0.5 and language_hint and language_hint.strip().lower() != "en" and attempt == 0:
                time.sleep(1)
                prompt_content = _format_interpretation_prompt(segments_text, language_hint) + "\n\nCRITICAL: The 'translation' field must contain only English. Translate each line to English; do not copy the original text into the translation field."
                last_lookup = lookup
                last_echo_ratio = echo_ratio
                continue
            # If this was a retry (we have last_lookup), return the result with fewer echoes
            if last_lookup is not None:
                return lookup if echo_ratio <= last_echo_ratio else last_lookup
            return lookup
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
                continue
            return last_lookup if last_lookup is not None else {}
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return last_lookup if last_lookup is not None else {}

    return last_lookup if last_lookup is not None else {}


def interpret_segments(segments: list, language_hint: str = None) -> list:
    """
    Interpret lyrics with robust retry strategy:
    1. Deduplicate unique lines
    2. Split into batches of ~25 lines; run batches in parallel (Phase 1)
    3. Gap-fill retry for missing lines; full retry if most failed
    4. Never throw away successful translations — graceful partial results
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = Anthropic()

    # Filter to segments with actual text
    text_segments = [s for s in segments if s['text'].strip()]

    if not text_segments:
        for seg in segments:
            seg['romanized'] = ''
            seg['translation'] = '(no lyrics detected)'
            seg['meaning'] = ''
        return segments

    # Deduplicate — only interpret unique lyrics
    unique_texts = []
    seen = set()
    for s in text_segments:
        text_lower = s['text'].strip().lower()
        if text_lower not in seen:
            unique_texts.append(s['text'])
            seen.add(text_lower)

    # ── Phase 1: Batch interpretation (parallel) ──
    batches = []
    for i in range(0, len(unique_texts), INTERPRETATION_BATCH_SIZE):
        batches.append(unique_texts[i:i + INTERPRETATION_BATCH_SIZE])

    interp_lookup = {}
    max_workers = min(len(batches), 6)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_interpret_batch, batch, client, language_hint): batch
            for batch in batches
        }
        for fut in as_completed(futures):
            batch_result = fut.result()
            interp_lookup.update(batch_result)

    # ── Phase 2: Gap-fill retry for missing lines (sequential) ──
    missing_texts = [t for t in unique_texts if t.strip().lower() not in interp_lookup]

    if missing_texts and len(missing_texts) <= len(unique_texts) * 0.5:
        retry_batch_size = min(15, INTERPRETATION_BATCH_SIZE)
        for i in range(0, len(missing_texts), retry_batch_size):
            retry_batch = missing_texts[i:i + retry_batch_size]
            time.sleep(1)
            retry_result = _interpret_batch(retry_batch, client, language_hint, max_retries=2)
            interp_lookup.update(retry_result)
    elif missing_texts and len(missing_texts) > len(unique_texts) * 0.5:
        time.sleep(2)
        full_retry_lookup = {}
        for batch in batches:
            batch_result = _interpret_batch(batch, client, language_hint, max_retries=2)
            full_retry_lookup.update(batch_result)
        for key, val in full_retry_lookup.items():
            if key not in interp_lookup:
                interp_lookup[key] = val

    # ── Phase 3: Apply interpretations to segments ──
    result = []
    for seg in segments:
        text_key = seg['text'].strip().lower()
        if text_key in interp_lookup:
            interp = interp_lookup[text_key]
            seg['romanized'] = interp.get('romanized', '')
            seg['translation'] = interp.get('translation', '')
            seg['meaning'] = interp.get('meaning', '')
        elif seg['text'].strip():
            # Line had text but interpretation permanently failed
            seg['romanized'] = ''
            seg['translation'] = seg['text']  # Fall back to original text
            seg['meaning'] = ''
        else:
            seg['romanized'] = ''
            seg['translation'] = ''
            seg['meaning'] = ''
        result.append(seg)

    return result


def _replace_known_hallucinations(segments: list) -> list:
    """Replace segments that exactly match known hallucination phrases (e.g. echoed prompt) with ♪."""
    out = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text in _KNOWN_HALLUCINATION_PHRASES:
            out.append({"start": seg["start"], "end": seg["end"], "text": "♪"})
        else:
            out.append(dict(seg))
    return out


def merge_early_repeated_hallucinations(segments: list, early_sec: float = 90.0, max_text_len: int = 50) -> list:
    """
    Replace known hallucination phrases (e.g. echoed prompt) with ♪, then merge runs of
    consecutive segments in the first part of the track that have the same short text (or ♪).
    Leaves long repeated phrases (e.g. chorus) unchanged.
    """
    if not segments:
        return segments
    segments = _replace_known_hallucinations(segments)
    result = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        start = seg.get("start", 0)
        if start > early_sec:
            result.append(dict(seg))
            i += 1
            continue
        text = (seg.get("text") or "").strip()
        text_lower = text.lower()
        # Merge only if same text is short or is ♪ (from known hallucinations)
        can_merge = len(text) <= max_text_len or text_lower == "♪"
        if not can_merge:
            result.append(dict(seg))
            i += 1
            continue
        j = i + 1
        while j < len(segments) and segments[j].get("start", 0) <= early_sec:
            t = (segments[j].get("text") or "").strip()
            tl = t.lower()
            if tl != text_lower or (len(t) > max_text_len and tl != "♪"):
                break
            j += 1
        if j > i + 1:
            result.append({
                "start": seg["start"],
                "end": segments[j - 1]["end"],
                "text": "♪",
            })
            i = j
        else:
            result.append(dict(seg))
            i += 1
    return result


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
    Use the actual end of the instrumental run (run_end) so we don't show lyrics
    ahead of the audio. Then ensure no segment starts before the previous ends
    (clip overlapping starts) so sync stays correct.
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
        result.append({
            'start': run_start,
            'end': run_end,
            'text': '♪',
            'romanized': '(instrumental)',
            'translation': 'Instrumental',
            'meaning': '',
        })
        i = j
    # Clip any segment that starts before the previous one ends (avoids lyrics ahead of audio)
    for k in range(1, len(result)):
        prev_end = result[k - 1]['end']
        if result[k]['start'] < prev_end:
            result[k] = dict(result[k])
            result[k]['start'] = prev_end
        if result[k]['start'] >= result[k]['end']:
            result[k]['end'] = result[k]['start'] + 0.01
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
            z-index: 10;
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
        // Scroll Streamlit app to top when player loads (e.g. after selecting a song)
        try {{ if (window.parent && window.parent !== window) window.parent.scrollTo({{ top: 0, left: 0, behavior: 'smooth' }}); }} catch (e) {{}}
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
                
                const useBins = Math.min(64, dataArray.length);
                const binsPerBar = useBins / NUM_BARS;
                // Bass floor: average of lowest bins so high-frequency bars (last few) still move with the beat
                let bassSum = 0;
                const bassRange = Math.min(8, dataArray.length);
                for (let b = 0; b < bassRange; b++) bassSum += dataArray[b];
                const bassFloor = bassSum / bassRange;
                // Overall energy (mid + high) so last bars get a bit more life when track is loud
                let energySum = 0;
                for (let b = 0; b < useBins; b++) energySum += dataArray[b];
                const energyAvg = energySum / useBins;
                
                waveBars.forEach((bar, i) => {{
                    const startBin = Math.floor(i * binsPerBar);
                    const endBin = Math.min(Math.floor((i + 1) * binsPerBar), useBins);
                    let value = 0;
                    for (let b = startBin; b < endBin; b++) {{
                        if (dataArray[b] > value) value = dataArray[b];
                    }}
                    // Stronger floor for last 8 bars (high-freq bins are usually quiet) so they move with the beat
                    const isLastEight = i >= NUM_BARS - 8;
                    const bassBlend = isLastEight ? 0.5 + (i - (NUM_BARS - 8)) / 8 * 0.35 : 0.35;
                    const energyBlend = isLastEight ? energyAvg * 0.25 : 0;
                    value = Math.max(value, bassFloor * bassBlend + energyBlend);
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
                    
                    // Auto-scroll on line change (resets userScrolledAway) — skip when in focus mode so the overlay doesn't move
                    userScrolledAway = false;
                    if (!focusMode) {{
                        scrollToLine(newIndex);
                    }}
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
                if (focusMode) {{
                    var wrapper = document.querySelector('.lyrics-wrapper');
                    if (wrapper) wrapper.scrollTo({{ top: 0, behavior: 'smooth' }});
                    if (currentLineIndex >= 0) updateFocusContent(currentLineIndex);
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

# Curated songs for landing carousel "See what Surasa does" (optional video_id for thumbnail)
CURATED_SONGS = {
    "🇫🇷 French": [
        {"title": "La Vie en Rose", "artist": "Édith Piaf", "query": "La Vie en Rose Edith Piaf official", "video_id": "CE5T3s7YPqc"},
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
        {"title": "Nel blu dipinto di blu (Volare)", "artist": "Domenico Modugno", "query": "Volare Domenico Modugno official", "video_id": "v2HKRAtUKNw"},
        {"title": "Con te partirò", "artist": "Andrea Bocelli", "query": "Con te partiro Andrea Bocelli official", "video_id": "-_B4A2yym8k"},
    ],
    "🇵🇹 Portuguese (Brazil)": [
        {"title": "Garota de Ipanema", "artist": "Antônio Carlos Jobim", "query": "Garota de Ipanema Jobim official", "video_id": "z4mNBMK5oK0"},
        {"title": "Ai se eu te pego", "artist": "Michel Teló", "query": "Ai se eu te pego Michel Telo official", "video_id": "hcm55lU9knw"},
    ],
    "🇩🇪 German": [
        {"title": "99 Luftballons", "artist": "Nena", "query": "99 Luftballons Nena official", "video_id": "Fpu5a0Bl8eY"},
        {"title": "Du hast", "artist": "Rammstein", "query": "Rammstein Du hast official", "video_id": "W3q8Od5qJio"},
    ],
    "🇮🇳 Hindi": [
        {"title": "Tum Hi Ho", "artist": "Arijit Singh", "query": "Tum Hi Ho Arijit Singh official", "video_id": "jr70FnJ4AGU"},
        {"title": "Kesariya", "artist": "Arijit Singh", "query": "Kesariya Brahmastra official", "video_id": "g6fnFALEseI"},
    ],
    "🇸🇦 Arabic": [
        {"title": "Habibi", "artist": "Mohamed Ramadan", "query": "Habibi Mohamed Ramadan official", "video_id": "KSqRXbynN2Q"},
        {"title": "Enta Eih", "artist": "Nancy Ajram", "query": "Nancy Ajram Enta Eih official", "video_id": "Cchy6MEoi6A"},
    ],
    "🇷🇺 Russian": [
        {"title": "Million Roses", "artist": "Alla Pugacheva", "query": "Million Roses Alla Pugacheva", "video_id": "RxtX0u01RqQ"},
        {"title": "Kalinka", "artist": "Traditional", "query": "Kalinka Russian folk official", "video_id": ""},
    ],
    "🇹🇷 Turkish": [
        {"title": "Gibi Gibi", "artist": "Sezen Aksu", "query": "Sezen Aksu Gibi Gibi official", "video_id": ""},
        {"title": "Düm Tek Tek", "artist": "Hadise", "query": "Dum Tek Tek Hadise Eurovision", "video_id": "tJURVrHy1C8"},
    ],
    "🇨🇳 Mandarin Chinese": [
        {"title": "Tian Mi Mi", "artist": "Teresa Teng", "query": "Tian Mi Mi Teresa Teng official", "video_id": "C_bbCsVFBNE"},
        {"title": "Qing Hua Ci", "artist": "Jay Chou", "query": "Qing Hua Ci Jay Chou official", "video_id": "nPTFcqhpRlc"},
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
        {"title": "See Tinh", "artist": "Hoang Thuy Linh", "query": "See Tinh Hoang Thuy Linh official", "video_id": "gZON_LiUvxc"},
        {"title": "Ghen", "artist": "Min & Erik", "query": "Ghen Min Erik official", "video_id": "QlYNB1NF9VE"},
    ],
    "🇮🇩 Indonesian": [
        {"title": "Kopi Dangdut", "artist": "Fahmi Shahab", "query": "Kopi Dangdut Fahmi Shahab official", "video_id": ""},
        {"title": "Lathi", "artist": "Weird Genius", "query": "Weird Genius Lathi official", "video_id": "zkc4JKVn_K8"},
    ],
    "🇵🇭 Tagalog": [
        {"title": "Buwan", "artist": "Juan Karlos", "query": "Buwan Juan Karlos official", "video_id": "NdCDxZVm42w"},
        {"title": "Hanggang", "artist": "Wency Cornejo", "query": "Hanggang Wency Cornejo official", "video_id": ""},
    ],
    "🇸🇪 Swedish": [
        {"title": "Dancing Queen", "artist": "ABBA", "query": "ABBA Dancing Queen official", "video_id": "xFrGuyw1V8s"},
        {"title": "Euphoria", "artist": "Loreen", "query": "Loreen Euphoria Eurovision official", "video_id": "bcnWysA9gxo"},
    ],
    "🇳🇱 Dutch": [
        {"title": "Venus", "artist": "Shocking Blue", "query": "Shocking Blue Venus official", "video_id": "BJ9zSrzIa7k"},
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
        {"title": "Diva", "artist": "Dana International", "query": "Dana International Diva Eurovision", "video_id": "4No1oClTp_E"},
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
        {"title": "Kolaveri Di", "artist": "Dhanush", "query": "Kolaveri Di Dhanush official", "video_id": "5DK-ZWyxZ8k"},
        {"title": "Jai Ho", "artist": "A.R. Rahman", "query": "Jai Ho Slumdog Millionaire official", "video_id": "xwwAVRyNmgQ"},
    ],
    "🇮🇳 Telugu": [
        {"title": "Naatu Naatu", "artist": "Rahul Sipligunj", "query": "Naatu Naatu RRR official", "video_id": "OsU0CGZoV8E"},
        {"title": "Bahubali", "artist": "M.M. Keeravani", "query": "Bahubali theme song official", "video_id": ""},
    ],
    "🇮🇳 Punjabi": [
        {"title": "Lemonade", "artist": "Diljit Dosanjh", "query": "Lemonade Diljit Dosanjh official", "video_id": "Sb9SsxBPBEU"},
        {"title": "High Rated Gabru", "artist": "Guru Randhawa", "query": "High Rated Gabru Guru Randhawa official", "video_id": "gzN5oGGo2vw"},
    ],
    "🇺🇦 Ukrainian": [
        {"title": "Stefania", "artist": "Kalush Orchestra", "query": "Stefania Kalush Eurovision official", "video_id": "lCerjLF8jlA"},
        {"title": "Chervona Ruta", "artist": "Sofia Rotaru", "query": "Chervona Ruta Sofia Rotaru", "video_id": "9OawC57_pjs"},
    ],
    "🇷🇴 Romanian": [
        {"title": "Dragostea din tei", "artist": "O-Zone", "query": "Dragostea din tei O-Zone official", "video_id": "8sl6fAllTfs"},
        {"title": "Stereo Love", "artist": "Edward Maya", "query": "Edward Maya Stereo Love official", "video_id": "p-Z3YrHJ1sU"},
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
        {"title": "Fairytale", "artist": "Alexander Rybak", "query": "Alexander Rybak Fairytale Eurovision", "video_id": "WXwgZL4zx9o"},
        {"title": "Take On Me", "artist": "a-ha", "query": "a-ha Take On Me official", "video_id": "djV11Xbc914"},
    ],
    "🇩🇰 Danish": [
        {"title": "Only Teardrops", "artist": "Emmelie de Forest", "query": "Only Teardrops Eurovision Denmark", "video_id": ""},
        {"title": "Smuk som et stjerneskud", "artist": "Medina", "query": "Medina Smuk som et stjerneskud", "video_id": ""},
    ],
    "🇫🇮 Finnish": [
        {"title": "Hard Rock Hallelujah", "artist": "Lordi", "query": "Lordi Hard Rock Hallelujah Eurovision", "video_id": "Njaju0owhbY"},
        {"title": "Sandstorm", "artist": "Darude", "query": "Darude Sandstorm official", "video_id": "erb4n8PW2qw"},
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
        {"title": "Fado Português", "artist": "Amália Rodrigues", "query": "Amalia Rodrigues Fado official", "video_id": "ARS7Zi-Zpkw"},
        {"title": "Amar pelos dois", "artist": "Salvador Sobral", "query": "Amar pelos dois Eurovision Portugal", "video_id": "Qotooj7ODCM"},
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
        {"title": "Habibi ya nour el ain", "artist": "Amr Diab", "query": "Amr Diab Habibi ya nour el ain", "video_id": "0Ma7ir9nwrY"},
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

# Related moods: suggestions can boost "similar mood" (e.g. melancholic → bittersweet)
MOOD_FAMILIES = [
    {"melancholic", "bittersweet", "nostalgic", "dreamy"},
    {"upbeat", "energetic", "joyful"},
    {"romantic", "peaceful"},
]
MAX_SUGGESTIONS_PER_ARTIST = 2  # Diversity: avoid 5 songs from the same artist
SEARCH_FALLBACK_MAX = 2  # Max suggestions to fill via YouTube search when thin


def _normalize_for_match(s: str) -> str:
    """Lowercase and strip for artist/language/mood comparison."""
    return (s or "").strip().lower()


def _mood_in_same_family(mood_a: str, mood_b: str) -> bool:
    """True if both moods belong to the same MOOD_FAMILIES group."""
    a, b = _normalize_for_match(mood_a), _normalize_for_match(mood_b)
    if not a or not b or a == b:
        return False
    for family in MOOD_FAMILIES:
        if a in family and b in family:
            return True
    return False


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
            elif mood_lower and _mood_in_same_family(mood_lower, song_mood):
                score += 1
                reasons.append("Similar mood")
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

    # ─── 4. Take up to max_suggestions, dedupe by _key, cap per-artist for diversity ───
    out = []
    artist_count = {}
    for c in candidates:
        if len(out) >= max_suggestions:
            break
        key_id = c.get("_key")
        if key_id in seen_keys:
            continue
        subtitle = (c.get("subtitle") or "").strip().lower()
        if subtitle and artist_count.get(subtitle, 0) >= MAX_SUGGESTIONS_PER_ARTIST:
            continue
        seen_keys.add(key_id)
        if subtitle:
            artist_count[subtitle] = artist_count.get(subtitle, 0) + 1
        item = {k: v for k, v in c.items() if not k.startswith("_")}
        out.append(item)

    # ─── 5. Fallback: fill with YouTube search when we have few suggestions ───
    need = max_suggestions - len(out)
    if need > 0 and SEARCH_FALLBACK_MAX > 0 and (channel or current_title):
        try:
            # Prefer "more from this artist" then "songs like this"
            query = f"{channel} songs" if channel else f"{current_title} official"
            limit = min(need, SEARCH_FALLBACK_MAX)
            search_results = search_youtube(query, max_results=limit + 3)
            added = 0
            for r in search_results:
                if added >= limit:
                    break
                url = r.get("url")
                title = r.get("title", "Unknown")
                ch = r.get("channel", "")
                if not url or url == current_url or url in seen_urls:
                    continue
                key_id = _song_key(title, ch)
                if key_id in seen_keys:
                    continue
                seen_urls.add(url)
                seen_keys.add(key_id)
                out.append({
                    "title": title,
                    "subtitle": ch,
                    "url": url,
                    "thumbnail": r.get("thumbnail", ""),
                    "type": "search",
                    "reason": "More like this" if channel else "Similar",
                })
                added += 1
        except Exception:
            pass

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

# Scroll to top when a new song is selected (consumed once per rerun)
if st.session_state.pop('_scroll_to_top', False):
    st.components.v1.html(
        "<script>try { window.parent.scrollTo({ top: 0, left: 0, behavior: 'smooth' }); } catch(e){} </script>",
        height=0,
    )

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
    st.components.v1.html(
        "<script>try { var w = window.parent && window.parent !== window ? window.parent : window; w.scrollTo({ top: 0, left: 0, behavior: 'smooth' }); } catch (e) {} </script>",
        height=0,
    )
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
    
    # Download song + Download lyrics + Choose another song (directly below player)
    st.caption("Download the audio or lyrics, or pick another song.")
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
    
    # Suggested next (similar songs)
    suggested = data.get('suggested_songs') or []
    if suggested:
        st.markdown("**You might also like**")
        st.caption("Same artist, similar mood, language, or more like this — one click to play")
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
                elif s.get('type') == 'search' and s.get('url'):
                    if st.button("▶ Play", key=f"sug_srch_{idx}", help="Play this song"):
                        st.session_state.pop('karaoke_data', None)
                        st.session_state['selected_url'] = s['url']
                        st.session_state['selected_title'] = s.get('title', 'Unknown')
                        st.session_state['auto_process'] = True
                        st.session_state['_scroll_to_top'] = True
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
                            st.session_state['_scroll_to_top'] = True
                            st.rerun()
        st.divider()

st.divider()

# Tabs: Search and History (curated songs are on the landing carousel)
tab1, tab2 = st.tabs(["🔍 Search", "📜 History"])

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
                        st.session_state['_scroll_to_top'] = True
                        st.rerun()
                st.divider()
        else:
            st.markdown("#### No results found")
            st.caption(f"We couldn't find anything for **\"{search_query}\"**. Try a different spelling, add the artist name, or paste a YouTube link below.")
    st.caption("Or paste a YouTube link")
    col_url, col_btn = st.columns([5, 1])
    with col_url:
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
            label_visibility="collapsed"
        )
    with col_btn:
        load_url_clicked = st.button("Load", key="load_youtube_url", type="primary", use_container_width=True)
    url_stripped = (youtube_url or "").strip()
    is_youtube = url_stripped and ("youtube.com" in url_stripped or "youtu.be" in url_stripped)
    if is_youtube and load_url_clicked:
        st.session_state.pop('karaoke_data', None)
        st.session_state['selected_url'] = url_stripped
        st.session_state['selected_title'] = "YouTube Video"
        st.session_state['auto_process'] = True
        st.session_state['_scroll_to_top'] = True
        st.rerun()

with tab2:
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
        st.markdown("""
<div style="text-align: center; padding: 2rem 1rem;">
    <div style="font-size: 3rem; margin-bottom: 0.5rem;">📜</div>
    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">Your listening history will appear here</div>
    <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem; max-width: 360px; margin: 0 auto;">
        Every song you play is saved for instant replay.<br>
        Head to <b>Search</b> or pick a song from the carousel above to get started!
    </div>
</div>
""", unsafe_allow_html=True)
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
                    _current_step = "Download"
                    update_progress(1, "Fetching audio from YouTube...", est_download)
                    download_messages = [
                        "Connecting to YouTube...",
                        "Downloading audio stream...",
                        "Converting to MP3...",
                    ]
                    with animated_status(detail_display, download_messages):
                        audio_path = download_audio(st.session_state['selected_url'], tmp_dir)
                    
                    # Step 2: Transcribe (auto-detect language)
                    _current_step = "Transcribe"
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
                        segments, detected_language = transcribe_with_timestamps(audio_path)
                    segments = merge_early_repeated_hallucinations(segments)
                    
                    # Count unique segments for optimization info
                    text_segments = [s for s in segments if s['text'].strip()]
                    unique_count = len(set(s['text'].strip().lower() for s in text_segments))

                    if not text_segments:
                        detail_display.caption("⚠️ No lyrics detected — this may be an instrumental track")
                    else:
                        detail_display.caption(f"✓ Found {len(text_segments)} lyric lines ({unique_count} unique)")
                    time_module.sleep(0.5)  # Brief pause to show the count
                    
                    # Step 3: Interpret with Claude Sonnet
                    _current_step = "Interpret"
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
                        interpreted_segments = interpret_segments(segments, language_hint=detected_language)
                        interpreted_segments = merge_instrumental_segments(interpreted_segments)

                    # Surface partial translation warnings
                    if text_segments:
                        translated_count = sum(
                            1 for s in interpreted_segments
                            if s['text'].strip() and (s.get('translation') or '').strip()
                               and s.get('translation') != s['text']
                        )
                        if translated_count < len(text_segments) * 0.5:
                            st.warning(f"⚠️ Only {translated_count}/{len(text_segments)} lines were translated. "
                                       "Some lines may show original text instead.")
                    
                    # Language, mood, and summary for badges, theme, and card
                    try:
                        lang, mood, summary = get_language_and_mood(interpreted_segments)
                    except Exception:
                        lang, mood, summary = "Unknown", "chill", ""
                    current_url = st.session_state['selected_url']
                    current_title = st.session_state.get('selected_title', 'Unknown')
                    channel = meta.get('channel')
                    try:
                        suggested_songs = get_suggested_songs(
                            lang, current_url, current_title,
                            mood=mood, channel=channel
                        )
                    except Exception:
                        suggested_songs = []
                    
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
                    import traceback
                    traceback.print_exc()
                    err_msg = str(e)[:200]
                    step_label = locals().get('_current_step', 'Unknown')
                    st.error(f"Something went wrong during **{step_label}**.")
                    st.caption(f"Details: {err_msg}")
                    if st.button("🔄 Try again", type="primary"):
                        st.session_state.pop('karaoke_data', None)
                        st.rerun()
    
    if 'selected_url' not in st.session_state:
        # ── How it works (compact, black background) ──
        _hw_steps = [
            ("🔍", "Search", "Find any song by name or paste a YouTube link"),
            ("🎤", "Transcribe", "AI listens and detects lyrics + language"),
            ("🔮", "Interpret", "Get translations, romanization, and cultural meaning"),
            ("🎶", "Play", "Karaoke mode syncs lyrics as you listen"),
        ]
        _hw_html = ""
        for _icon, _label, _desc in _hw_steps:
            _hw_html += f'<div style="flex:1;text-align:center;padding:16px 8px;"><div style="font-size:2rem;">{_icon}</div><div style="font-weight:600;font-size:0.95rem;margin:6px 0;">{_label}</div><div style="color:rgba(0,0,0,0.55);font-size:0.82rem;line-height:1.4;">{_desc}</div></div>'
        st.markdown(
            f'<div style="background:#f0f2f6;border-radius:14px;padding:24px 16px;margin-bottom:1.5rem;">'
            f'<div style="text-align:center;margin-bottom:12px;"><div style="font-weight:600;font-size:1.1rem;">How it works</div></div>'
            f'<div style="display:flex;gap:8px;justify-content:center;">{_hw_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── "Try it" demo carousel (all curated songs) ──
        _demo_songs_flat = []
        for _lang, _songs in CURATED_SONGS.items():
            for _s in _songs:
                _demo_songs_flat.append({**_s, "lang": _lang})
        _try_idx = None
        try:
            _try_idx = st.query_params.get("try")
        except Exception:
            pass
        if _try_idx is not None:
            try:
                _idx = int(_try_idx)
                if 0 <= _idx < len(_demo_songs_flat):
                    _ds = _demo_songs_flat[_idx]
                    with st.spinner("Finding video..."):
                        _res = search_youtube(_ds["query"])
                    if _res:
                        st.session_state.pop("karaoke_data", None)
                        st.session_state["selected_url"] = _res[0]["url"]
                        st.session_state["selected_title"] = _res[0]["title"]
                        st.session_state["auto_process"] = True
                        st.session_state["_scroll_to_top"] = True
                        try:
                            del st.query_params["try"]
                        except Exception:
                            pass
                        st.rerun()
            except (ValueError, TypeError):
                pass
        st.markdown("#### See what Surasa does")
        st.caption("Pick a song to experience AI-powered lyrics, translations, and cultural context. Scroll for more.")
        _cards_html = []
        for _di, _ds in enumerate(_demo_songs_flat):
            _vid = _ds.get("video_id") or ""
            _thumb = _youtube_thumbnail_url(_vid) if _vid else ""
            if _thumb:
                _img = f'<img src="{html.escape(_thumb)}" alt="" style="width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:8px;display:block;" />'
            else:
                _flag = (_ds.get("lang") or "🎵").split(" ")[0]
                _img = f'<div style="width:100%;aspect-ratio:16/9;border-radius:8px;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;font-size:2rem;">{_flag}</div>'
            _cards_html.append(
                f'<div class="surasa-demo-card" style="flex:0 0 200px;min-width:200px;">'
                f'<a href="?try={_di}" target="_top" style="text-decoration:none;color:inherit;display:block;border:1px solid rgba(0,0,0,0.08);border-radius:12px;padding:10px;background:#f0f2f6;height:100%;">'
                f'{_img}'
                f'<div style="margin-top:8px;"><div style="font-weight:600;font-size:0.9rem;">{html.escape(_ds["title"])}</div>'
                f'<div style="color:rgba(0,0,0,0.5);font-size:0.78rem;">{html.escape(_ds["artist"])} · {html.escape(_ds["lang"])}</div></div>'
                f'<div style="margin-top:8px;font-size:0.75rem;color:rgba(0,0,0,0.65);">▶ Try this</div>'
                f'</a></div>'
            )
        # Infinite carousel: rotate DOM elements on arrow click
        _carousel_html = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"></head><body style="margin:0;padding:0;overflow:hidden;">
<div style="display:flex;align-items:center;gap:8px;margin:8px 0;">
<button type="button" id="carousel-prev" aria-label="Scroll left" style="flex-shrink:0;width:40px;height:40px;border-radius:50%;border:1px solid rgba(0,0,0,0.15);background:#f0f2f6;cursor:pointer;font-size:1.2rem;display:flex;align-items:center;justify-content:center;">‹</button>
<div id="carousel-track" style="overflow:hidden;flex:1;padding:8px 0;">
<div id="carousel-inner" style="display:flex;gap:12px;transition:transform 0.35s ease;">
{''.join(_cards_html)}
</div>
</div>
<button type="button" id="carousel-next" aria-label="Scroll right" style="flex-shrink:0;width:40px;height:40px;border-radius:50%;border:1px solid rgba(0,0,0,0.15);background:#f0f2f6;cursor:pointer;font-size:1.2rem;display:flex;align-items:center;justify-content:center;">›</button>
</div>
<script>
(function() {{
  var inner = document.getElementById('carousel-inner');
  var prev = document.getElementById('carousel-prev');
  var next = document.getElementById('carousel-next');
  if (!inner || !prev || !next) return;
  var moving = false;
  function getCardWidth() {{
    var card = inner.children[0];
    if (!card) return 212;
    var style = getComputedStyle(card);
    return card.offsetWidth + parseInt(style.marginRight || 0) + 12;
  }}
  next.addEventListener('click', function() {{
    if (moving) return;
    moving = true;
    var w = getCardWidth();
    inner.style.transition = 'transform 0.35s ease';
    inner.style.transform = 'translateX(-' + w + 'px)';
    inner.addEventListener('transitionend', function handler() {{
      inner.removeEventListener('transitionend', handler);
      inner.style.transition = 'none';
      inner.style.transform = 'translateX(0)';
      inner.appendChild(inner.children[0]);
      moving = false;
    }});
  }});
  prev.addEventListener('click', function() {{
    if (moving) return;
    moving = true;
    var w = getCardWidth();
    inner.style.transition = 'none';
    inner.insertBefore(inner.children[inner.children.length - 1], inner.children[0]);
    inner.style.transform = 'translateX(-' + w + 'px)';
    requestAnimationFrame(function() {{
      requestAnimationFrame(function() {{
        inner.style.transition = 'transform 0.35s ease';
        inner.style.transform = 'translateX(0)';
        inner.addEventListener('transitionend', function handler() {{
          inner.removeEventListener('transitionend', handler);
          moving = false;
        }});
      }});
    }});
  }});
}})();
</script>
</body></html>
"""
        st.components.v1.html(_carousel_html, height=260, scrolling=False)

st.divider()
st.caption("© 2026 Abhinav Deshmukh · Lyrics and interpretations are AI-generated; use for learning only.")