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
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from streamlit_searchbox import st_searchbox

load_dotenv()

# Simple file-based cache for processed songs
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(url: str, language: str) -> str:
    """Generate cache key from URL and language."""
    return hashlib.md5(f"{url}:{language}".encode()).hexdigest()

def get_cached_result(url: str, language: str) -> dict:
    """Try to get cached result for a song."""
    cache_key = get_cache_key(url, language)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def save_to_cache(url: str, language: str, data: dict):
    """Save processed song to cache."""
    cache_key = get_cache_key(url, language)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    except:
        pass  # Fail silently

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

SEGMENT_INTERPRETATION_PROMPT = """Translate and romanize these song lyrics. Return ONLY a JSON array, no other text.

For each line, provide:
- original: the exact original text
- romanized: phonetic pronunciation in English letters (e.g., "مرحبا" → "marhaba")
- translation: English translation (poetic, under 15 words)
- meaning: brief cultural/emotional context (under 20 words)

Return format - ONLY this JSON array, nothing else:
[
  {{"original": "...", "romanized": "...", "translation": "...", "meaning": "..."}},
  ...
]

Lyrics:
{segments}
"""

def search_youtube(query: str, max_results: int = 5) -> list:
    """Search YouTube and return list of results."""
    try:
        result = subprocess.run(
            ["yt-dlp", f"ytsearch{max_results}:{query}", "--dump-json", "--flat-playlist"],
            capture_output=True, text=True, timeout=30
        )
        
        results = []
        for line in result.stdout.strip().split('\n'):
            if line:
                data = json.loads(line)
                results.append({
                    'title': data.get('title', 'Unknown'),
                    'url': f"https://www.youtube.com/watch?v={data.get('id', '')}",
                    'channel': data.get('channel', data.get('uploader', 'Unknown')),
                    'duration': data.get('duration_string', ''),
                })
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

def transcribe_with_timestamps(audio_path: str, language: str = None) -> list:
    """Transcribe audio with word-level timestamps."""
    import time
    client = OpenAI()
    
    # Retry logic for connection errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as audio_file:
                # Build API parameters
                params = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                
                # Add language hint if specified
                if language and language != "auto":
                    params["language"] = language
                
                transcript = client.audio.transcriptions.create(**params)
            
            # Extract segments with timestamps
            segments = []
            for seg in transcript.segments:
                segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip()
                })
            
            return segments
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                raise e

# Supported languages for Whisper
LANGUAGE_OPTIONS = {
    "auto": "🌐 Auto-detect",
    "ar": "🇸🇦 Arabic (العربية)",
    "hi": "🇮🇳 Hindi (हिन्दी)",
    "ur": "🇵🇰 Urdu (اردو)",
    "pa": "🇮🇳 Punjabi (ਪੰਜਾਬੀ)",
    "bn": "🇧🇩 Bengali (বাংলা)",
    "ta": "🇮🇳 Tamil (தமிழ்)",
    "te": "🇮🇳 Telugu (తెలుగు)",
    "fa": "🇮🇷 Persian (فارسی)",
    "tr": "🇹🇷 Turkish (Türkçe)",
    "es": "🇪🇸 Spanish (Español)",
    "fr": "🇫🇷 French (Français)",
    "ru": "🇷🇺 Russian (Русский)",
    "ko": "🇰🇷 Korean (한국어)",
    "ja": "🇯🇵 Japanese (日本語)",
    "zh": "🇨🇳 Chinese (中文)",
}

def interpret_segments(segments: list) -> list:
    """Get interpretation for each segment using Claude (optimized - skips duplicates)."""
    import time
    client = Anthropic()
    
    # Filter to segments with actual text
    text_segments = [s for s in segments if s['text'].strip()]
    
    if not text_segments:
        # No text segments - mark all as instrumental
        for seg in segments:
            seg['romanized'] = ''
            seg['translation'] = '(no lyrics detected)'
            seg['meaning'] = ''
        return segments
    
    # OPTIMIZATION: Deduplicate - only interpret unique lyrics
    unique_texts = []
    seen = set()
    for s in text_segments:
        text_lower = s['text'].strip().lower()
        if text_lower not in seen:
            unique_texts.append(s['text'])
            seen.add(text_lower)
    
    # Prepare unique segments text
    segments_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(unique_texts)])
    
    # Retry logic for connection errors
    max_retries = 3
    response = None
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Faster than Sonnet
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": SEGMENT_INTERPRETATION_PROMPT.format(segments=segments_text)
                    }
                ]
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                # Return segments without interpretation on failure
                for seg in segments:
                    seg['romanized'] = ''
                    seg['translation'] = '(connection error - try again)'
                    seg['meaning'] = ''
                return segments
    
    # Parse JSON response
    response_text = response.content[0].text
    
    # Extract JSON from response (handle markdown code blocks)
    try:
        json_text = response_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]
        
        # Try to find JSON array in the response
        json_text = json_text.strip()
        if not json_text.startswith('['):
            # Try to find the array in the text
            start = json_text.find('[')
            end = json_text.rfind(']') + 1
            if start >= 0 and end > start:
                json_text = json_text[start:end]
        
        interpretations = json.loads(json_text)
    except (json.JSONDecodeError, Exception) as e:
        # If JSON parsing fails, show the error in the first segment
        error_msg = f"Parse error: {str(e)[:100]}"
        preview = response_text[:200] if response_text else "Empty response"
        
        for i, seg in enumerate(segments):
            seg['romanized'] = ''
            if i == 0:
                seg['translation'] = f'⚠️ {error_msg}'
                seg['meaning'] = f'Response preview: {preview}...'
            else:
                seg['translation'] = '(see first segment for error)'
                seg['meaning'] = ''
        return segments
    
    # Build lookup from unique texts to interpretations
    interp_lookup = {}
    for i, text in enumerate(unique_texts):
        if i < len(interpretations):
            interp_lookup[text.strip().lower()] = interpretations[i]
    
    # Apply interpretations to ALL segments (including duplicates)
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
    
    return result

def get_audio_base64(audio_path: str) -> str:
    """Convert audio file to base64 for embedding."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def create_karaoke_player(audio_base64: str, segments: list, audio_format: str = "mp3") -> str:
    """Create HTML/JS karaoke player."""
    
    # Convert segments to JSON for JavaScript
    segments_json = json.dumps(segments)
    
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        .karaoke-container {{
            font-family: 'Inter', sans-serif;
            max-width: 100%;
            margin: 0 auto;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 24px;
            color: white;
        }}
        
        .audio-controls {{
            margin-bottom: 20px;
        }}
        
        .audio-controls audio {{
            width: 100%;
            border-radius: 8px;
        }}
        
        .lyrics-container {{
            height: 400px;
            overflow-y: auto;
            scroll-behavior: smooth;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }}
        
        .lyric-segment {{
            padding: 16px;
            margin: 8px 0;
            border-radius: 8px;
            transition: all 0.3s ease;
            opacity: 0.4;
            border-left: 3px solid transparent;
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
    </style>
    
    <div class="karaoke-container">
        <div class="audio-controls">
            <audio id="audioPlayer" controls>
                <source src="data:audio/{audio_format};base64,{audio_base64}" type="audio/{audio_format}">
            </audio>
            <div class="progress-info">
                <span id="currentSegment">Ready to play</span>
                <span id="timeDisplay">0:00 / 0:00</span>
            </div>
        </div>
        
        <div class="lyrics-container" id="lyricsContainer">
        </div>
    </div>
    
    <script>
        const segments = {segments_json};
        const audio = document.getElementById('audioPlayer');
        const container = document.getElementById('lyricsContainer');
        const currentSegmentDisplay = document.getElementById('currentSegment');
        const timeDisplay = document.getElementById('timeDisplay');
        
        // Build lyrics HTML
        let lyricsHTML = '';
        segments.forEach((seg, idx) => {{
            if (seg.text) {{
                const romanized = seg.romanized || '';
                const translation = seg.translation || '';
                const meaning = seg.meaning || '';
                lyricsHTML += `
                    <div class="lyric-segment" id="segment-${{idx}}" data-start="${{seg.start}}" data-end="${{seg.end}}">
                        <div class="time-badge">${{formatTime(seg.start)}}</div>
                        <div class="original">${{seg.text}}</div>
                        ${{romanized ? `<div class="romanized">${{romanized}}</div>` : ''}}
                        <div class="translation">${{translation || '(translating...)'}}</div>
                        ${{meaning ? `<div class="meaning">${{meaning}}</div>` : ''}}
                    </div>
                `;
            }}
        }});
        container.innerHTML = lyricsHTML;
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Sync lyrics with audio
        audio.addEventListener('timeupdate', () => {{
            const currentTime = audio.currentTime;
            const duration = audio.duration || 0;
            timeDisplay.textContent = `${{formatTime(currentTime)}} / ${{formatTime(duration)}}`;
            
            let activeIdx = -1;
            segments.forEach((seg, idx) => {{
                const el = document.getElementById(`segment-${{idx}}`);
                if (!el) return;
                
                if (currentTime >= seg.start && currentTime < seg.end) {{
                    el.classList.add('active');
                    el.classList.remove('past');
                    activeIdx = idx;
                    
                    // Scroll into view
                    el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }} else if (currentTime >= seg.end) {{
                    el.classList.remove('active');
                    el.classList.add('past');
                }} else {{
                    el.classList.remove('active', 'past');
                }}
            }});
            
            if (activeIdx >= 0) {{
                currentSegmentDisplay.textContent = `Line ${{activeIdx + 1}} of ${{segments.length}}`;
            }}
        }});
        
        // Click on segment to seek
        document.querySelectorAll('.lyric-segment').forEach(el => {{
            el.addEventListener('click', () => {{
                const start = parseFloat(el.dataset.start);
                audio.currentTime = start;
                audio.play();
            }});
        }});
    </script>
    """
    
    return html

# Page config
st.set_page_config(
    page_title="Surasa",
    page_icon="🎶",
    layout="wide"
)

# Header
st.title("🎶 Surasa")
st.markdown("*सुर + रस — The essence of melody*")

# Check if we have karaoke data to display (song is ready)
has_karaoke = 'karaoke_data' in st.session_state

# If song is ready, show karaoke player FIRST (front and center)
if has_karaoke:
    st.markdown(f"### 🎤 {st.session_state.get('selected_title', 'Now Playing')}")
    st.caption("Click any line to jump to that part of the song")
    
    data = st.session_state['karaoke_data']
    karaoke_html = create_karaoke_player(
        data['audio_base64'],
        data['segments'],
        data['audio_format']
    )
    
    st.components.v1.html(karaoke_html, height=550, scrolling=False)
    
    # Button to search for another song
    st.divider()
    if st.button("🎶 Discover Another Song", use_container_width=True):
        for key in ['selected_url', 'selected_title', 'karaoke_data']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

else:
    # Show search interface only when no song is playing
    
    # Processing section (appears at top when processing)
    processing_container = st.container()
    
    # Check if we're currently processing - show that first
    if 'selected_url' in st.session_state and 'karaoke_data' not in st.session_state:
        with processing_container:
            st.info(f"🎵 Processing: **{st.session_state.get('selected_title', 'Song')}**")
    
    st.divider()
    
    # Language selector (above tabs)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Select the song's language for better transcription accuracy:")
    with col2:
        selected_language = st.selectbox(
            "Language",
            options=list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda x: LANGUAGE_OPTIONS[x],
            index=0,
            label_visibility="collapsed"
        )
    
    # Input method tabs
    tab1, tab2 = st.tabs(["🔍 Search Song", "🔗 YouTube Link"])
    
    with tab1:
        # Autocomplete search box
        search_query = st_searchbox(
            get_youtube_suggestions,
            key="song_search",
            placeholder="Search for a song (e.g., Pasoori, Tum Hi Ho...)",
            clear_on_submit=False,
        )
        
        if search_query:
            with st.spinner("Finding songs..."):
                results = search_youtube(search_query)
            
            if results:
                st.markdown("### Select a song:")
                for i, result in enumerate(results):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{result['title']}**")
                        st.caption(f"{result['channel']} • {result['duration']}")
                    with col2:
                        if st.button("▶️ Play", key=f"select_{i}"):
                            st.session_state['selected_url'] = result['url']
                            st.session_state['selected_title'] = result['title']
                            st.session_state['auto_process'] = True
                            st.rerun()
                    st.divider()
    
    with tab2:
        youtube_url = st.text_input(
            "Paste YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input"
        )
        
        # Auto-process when URL is entered
        if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
            st.session_state['selected_url'] = youtube_url
            st.session_state['selected_title'] = "YouTube Video"
            st.session_state['auto_process'] = True
    
    # Process selected song
    if 'selected_url' in st.session_state:
        # Auto-process if no karaoke data yet
        should_process = 'karaoke_data' not in st.session_state
        
        if should_process:
            # Check cache first
            cached = get_cached_result(st.session_state['selected_url'], selected_language)
            if cached:
                with processing_container:
                    st.success("⚡ Loaded from cache!")
                st.session_state['karaoke_data'] = cached
                st.rerun()
            
            # Show processing status at the TOP (in the container we created earlier)
            with processing_container:
                # Create temp directory
                tmp_dir = tempfile.mkdtemp()
                
                try:
                    with st.status("Working on it...", expanded=True) as status:
                        st.write("⬇️ Downloading audio...")
                        audio_path = download_audio(st.session_state['selected_url'], tmp_dir)
                        
                        lang_name = LANGUAGE_OPTIONS.get(selected_language, "Auto-detect")
                        st.write(f"🎤 Transcribing ({lang_name})...")
                        segments = transcribe_with_timestamps(audio_path, language=selected_language)
                        
                        # Count unique segments for optimization info
                        text_segments = [s for s in segments if s['text'].strip()]
                        unique_count = len(set(s['text'].strip().lower() for s in text_segments))
                        st.write(f"📝 Found {len(text_segments)} segments ({unique_count} unique)")
                        
                        st.write(f"🔮 Interpreting {unique_count} unique segments...")
                        interpreted_segments = interpret_segments(segments)
                        
                        st.write("🎨 Building karaoke player...")
                        audio_base64 = get_audio_base64(audio_path)
                        
                        # Determine audio format
                        audio_ext = os.path.splitext(audio_path)[1].lstrip('.')
                        if audio_ext == 'webm':
                            audio_format = 'webm'
                        else:
                            audio_format = 'mpeg'
                        
                        status.update(label="✅ Ready!", state="complete", expanded=False)
                    
                    # Store data for display
                    karaoke_data = {
                        'audio_base64': audio_base64,
                        'segments': interpreted_segments,
                        'audio_format': audio_format
                    }
                    st.session_state['karaoke_data'] = karaoke_data
                    
                    # Save to cache for next time
                    save_to_cache(st.session_state['selected_url'], selected_language, karaoke_data)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Show welcome message if no song selected and no search
    if 'selected_url' not in st.session_state:
        st.info("👆 Search for a song or paste a YouTube link")
        
        st.markdown("### How it works")
        st.markdown("""
        1. **Search** for any song or paste a YouTube link
        2. **AI transcribes** the lyrics with timestamps
        3. **AI interprets** each line's meaning
        4. **Karaoke mode** syncs lyrics as the song plays
        
        Perfect for:
        - 🎶 Understanding songs in languages you don't speak
        - 📚 Discovering cultural references and metaphors  
        - 💭 Following along with synchronized lyrics
        """)
