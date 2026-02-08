"""
Surasa - The Song Meaning App
From melody to meaning, in any language.

Run with: streamlit run app.py
"""

import base64
import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Generator

import streamlit as st
from dotenv import load_dotenv
from streamlit_searchbox import st_searchbox

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from surasa.config.settings import settings, CURATED_SONGS, MOOD_THEMES
from surasa.utils import (
    search_youtube,
    get_video_duration,
    download_audio,
    get_youtube_suggestions,
    get_cached_result,
    save_to_cache,
    get_cached_songs,
    clear_cache,
)
from surasa.services import (
    transcribe_with_timestamps,
    interpret_segments,
    get_similar_songs,
)
from surasa.templates import (
    render_karaoke_player,
    render_shareable_karaoke,
    render_animated_status,
    render_skeleton_loading,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


# =============================================================================
# Rate Limiting
# =============================================================================

DAILY_LIMIT = 10  # Max songs per session per day

def check_rate_limit() -> tuple[bool, int]:
    """
    Check if user has exceeded daily rate limit.
    Returns (is_allowed, remaining_count).
    """
    from datetime import date
    
    today = date.today().isoformat()
    
    # Initialize rate limit tracking
    if 'rate_limit_date' not in st.session_state:
        st.session_state['rate_limit_date'] = today
        st.session_state['rate_limit_count'] = 0
    
    # Reset count if it's a new day
    if st.session_state['rate_limit_date'] != today:
        st.session_state['rate_limit_date'] = today
        st.session_state['rate_limit_count'] = 0
    
    remaining = DAILY_LIMIT - st.session_state['rate_limit_count']
    return remaining > 0, remaining


def increment_rate_limit():
    """Increment the rate limit counter after processing a song."""
    st.session_state['rate_limit_count'] = st.session_state.get('rate_limit_count', 0) + 1


# =============================================================================
# Helper Functions
# =============================================================================

def estimate_transcription_and_interpretation_sec(audio_duration_sec: float) -> tuple[float, float]:
    """
    Estimate time for Whisper transcription + Claude interpretation from audio duration.
    Returns (low_sec, high_sec) for a range (e.g. "about 1–2 min").
    """
    mins = audio_duration_sec / 60.0
    # Whisper: ~15–30 sec per minute of audio; interpretation: ~5–15 sec per minute (by line count proxy)
    low_per_min = 20
    high_per_min = 45
    low_sec = max(15, mins * low_per_min)
    high_sec = max(30, mins * high_per_min)
    return (low_sec, high_sec)


def format_time_estimate(low_sec: float, high_sec: float) -> str:
    """Format estimate as user-facing string, e.g. 'about 30–90 sec' or 'about 1–2 min'."""
    if high_sec < 60:
        return f"about {int(low_sec)}–{int(high_sec)} sec"
    low_min = low_sec / 60
    high_min = high_sec / 60
    if low_min < 1 and high_min < 1:
        return f"about {int(low_sec)}–{int(high_sec)} sec"
    if low_min < 1:
        return f"about 30 sec–{int(round(high_min))} min"
    return f"about {int(round(low_min))}–{int(round(high_min))} min"


def get_audio_base64(audio_path: str) -> str:
    """Convert audio file to base64 for embedding."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@contextmanager
def animated_status(placeholder, messages: list, interval: float = 2.0) -> Generator:
    """
    Show rotating status messages while a long operation runs.
    Uses client-side JS so it works even when Python is blocked.
    """
    html = render_animated_status(messages, int(interval * 1000))
    placeholder.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


def show_skeleton_loading(placeholder, num_lines: int = 5):
    """Show a subtle skeleton loading UI."""
    with placeholder.container():
        st.markdown(render_skeleton_loading(num_lines), unsafe_allow_html=True)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Surasa",
    page_icon="🎶",
    layout="wide"
)

# When song ends in player, it redirects here with ?new_search=1 → show search + suggested songs
if st.query_params.get('new_search'):
    st.query_params.clear()
    # Preserve similar songs and title so we can show "You might also like" on the search view
    if 'karaoke_data' in st.session_state:
        data = st.session_state['karaoke_data']
        st.session_state['post_play_similar_songs'] = data.get('similar_songs', [])
        st.session_state['post_play_title'] = st.session_state.get('selected_title', '')
    for key in ['selected_url', 'selected_title', 'karaoke_data']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Handle similar song search from celebration overlay
similar_search_query = st.query_params.get('similar_search', None)
if similar_search_query:
    st.query_params.clear()
    results = search_youtube(similar_search_query)
    if results:
        st.session_state['selected_url'] = results[0]['url']
        st.session_state['selected_title'] = results[0]['title']
        if 'karaoke_data' in st.session_state:
            del st.session_state['karaoke_data']
        st.rerun()


# =============================================================================
# Header
# =============================================================================

st.title("🎶 Surasa")
st.markdown("*From melody to meaning, in any language*")

# Show rate limit status (subtle)
_, remaining = check_rate_limit()
if remaining <= 3:
    st.caption(f"⚡ {remaining} songs remaining today")


# =============================================================================
# Main Content – search bar always at top, then player or pipeline below
# =============================================================================

has_karaoke = 'karaoke_data' in st.session_state

# When returning after song end, scroll to top so the search bar is visible
if st.session_state.get('post_play_similar_songs') or st.session_state.get('post_play_title'):
    st.components.v1.html(
        "<script>try { window.parent.scrollTo(0,0); } catch(e) { window.scrollTo(0,0); }</script>",
        height=0,
    )
# Sticky header + search tabs so they stay visible when scrolling
st.markdown("""
<style>
/* Tighter gap between tagline and search tabs */
.block-container > div:nth-child(1),
.block-container > div:nth-child(2),
.block-container > div:nth-child(3) { padding-bottom: 0.1rem !important; margin-bottom: 0 !important; }
.block-container hr { margin-top: 0.25rem !important; margin-bottom: 0.25rem !important; }
.block-container > div:nth-child(1) { position: sticky !important; top: 0 !important; z-index: 999 !important; background: var(--background-color) !important; }
.block-container > div:nth-child(2) { position: sticky !important; top: 3.5rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(3) { position: sticky !important; top: 5.5rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(4) { position: sticky !important; top: 7rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(5) { position: sticky !important; top: 8.5rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(6) { position: sticky !important; top: 10rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(7) { position: sticky !important; top: 11.5rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(8) { position: sticky !important; top: 13rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(9) { position: sticky !important; top: 14.5rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
.block-container > div:nth-child(10) { position: sticky !important; top: 16rem !important; z-index: 999 !important; background: var(--background-color) !important; padding-bottom: 0.25rem !important; }
</style>
""", unsafe_allow_html=True)

st.divider()

# Processing bar MUST be above "Select a song" – show it here, before the tabs
processing_container = st.container()
progress_bar = step_display = detail_display = time_display = None
if (
    'selected_url' in st.session_state
    and 'karaoke_data' not in st.session_state
    and st.session_state.get('auto_process')
):
    with processing_container:
        st.info(f"🎵 Processing: **{st.session_state.get('selected_title', 'Song')}**")
        progress_bar = st.progress(0)
        step_display = st.empty()
        detail_display = st.empty()
        time_display = st.empty()
elif 'selected_url' in st.session_state and 'karaoke_data' not in st.session_state:
    with processing_container:
        st.info(f"🎵 Processing: **{st.session_state.get('selected_title', 'Song')}**")

# Search bar + tabs (Select a song appears inside Search tab, below this)
tab1, tab2, tab3 = st.tabs(["🔍 Search", "🌍 Browse by Language", "📚 History"])

with tab1:
    search_query = st_searchbox(
        get_youtube_suggestions,
        key="song_search",
        placeholder="Search any song (e.g., La Vie en Rose, Despacito, Gangnam Style...)",
        clear_on_submit=False,
    )
    if search_query and 'selected_url' not in st.session_state:
        with st.spinner("Finding songs..."):
            results = search_youtube(search_query)
        if results:
            st.markdown("### Select a song:")
            for i, result in enumerate(results):
                col_thumb, col_info = st.columns([1, 6])
                with col_thumb:
                    if result.get('thumbnail'):
                        st.image(result['thumbnail'], width=settings.ui.thumbnail_width)
                with col_info:
                    if st.button(
                        f"▶️  **{result['title']}**\n\n{result['channel']} • {result['duration']}",
                        key=f"select_{i}",
                        use_container_width=True
                    ):
                        st.session_state['selected_url'] = result['url']
                        st.session_state['selected_title'] = result['title']
                        st.session_state['auto_process'] = True
                        st.rerun()

with tab2:
    youtube_url = st.text_input(
        "Paste YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url_input"
    )
    if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
        st.session_state['selected_url'] = youtube_url
        st.session_state['selected_title'] = "YouTube Video"
        st.session_state['auto_process'] = True

with tab3:
    st.markdown("### Discover songs by language")
    st.caption("Curated classics and hits — click to play instantly")
    selected_language = st.selectbox(
        "Choose a language",
        options=list(CURATED_SONGS.keys()),
        index=0,
        key="browse_language"
    )
    if selected_language:
        songs = CURATED_SONGS.get(selected_language, [])
        for i, song in enumerate(songs):
            col_icon, col_info = st.columns([1, 6])
            with col_icon:
                st.markdown(
                    f'<div style="width:80px;height:60px;background:linear-gradient(135deg,#1a1a2e,#16213e);'
                    f'border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:24px;">🎵</div>',
                    unsafe_allow_html=True
                )
            with col_info:
                if st.button(
                    f"▶️  **{song['title']}**\n\nby {song['artist']}",
                    key=f"curated_{selected_language}_{i}",
                    use_container_width=True
                ):
                    with st.spinner("Finding best version..."):
                        results = search_youtube(song['query'], max_results=1)
                    if results:
                        st.session_state['selected_url'] = results[0]['url']
                        st.session_state['selected_title'] = f"{song['title']} - {song['artist']}"
                        st.session_state['auto_process'] = True
                        st.rerun()
                    else:
                        st.error("Could not find this song on YouTube")

with tab3:
    cached_songs = get_cached_songs()
    if cached_songs:
        st.markdown("### Previously played songs")
        st.caption("Click to replay instantly (no processing needed)")
        for i, song in enumerate(cached_songs):
            col_thumb, col_info = st.columns([1, 6])
            with col_thumb:
                if song.get('thumbnail'):
                    st.image(song['thumbnail'], width=settings.ui.thumbnail_width)
            with col_info:
                if st.button(
                    f"▶️  **{song['title']}**\n\nPlayed: {song['cached_at']}",
                    key=f"history_{i}",
                    use_container_width=True
                ):
                    st.session_state['selected_url'] = song['url']
                    st.session_state['selected_title'] = song['title']
                    cached_data = get_cached_result(song['url'], "auto")
                    if cached_data:
                        st.session_state['karaoke_data'] = cached_data
                    st.rerun()
        st.divider()
        if st.button("🗑️ Clear History", key="clear_history"):
            try:
                deleted = clear_cache()
                st.success(f"Cleared {deleted} cached songs")
            except Exception as e:
                st.error(f"Failed to clear history: {e}")
            st.rerun()
    else:
        st.info("No songs in history yet. Search for a song to get started!")

# Below the search bar: show player when playing, else progress + "You might also like" + pipeline
if has_karaoke:
    # -------------------------------------------------------------------------
    # Karaoke Player (search bar remains above)
    # -------------------------------------------------------------------------
    st.markdown(f"### 🎤 {st.session_state.get('selected_title', 'Now Playing')}")
    st.caption("Click any line to jump • **Shortcuts:** Space = play/pause, ←→ = skip 5s, F = focus mode")

    data = st.session_state['karaoke_data']
    mood = data.get('mood', 'playful')
    summary = data.get('summary', '')
    similar_songs = data.get('similar_songs', [])

    karaoke_html = render_karaoke_player(
        data['audio_base64'],
        data['segments'],
        data['audio_format'],
        mood,
        summary,
        similar_songs
    )

    st.components.v1.html(karaoke_html, height=settings.ui.player_height, scrolling=True)

    if similar_songs:
        st.markdown("#### 🎵 You might also like")
        cols = st.columns(min(len(similar_songs), settings.ui.max_similar_songs))
        for idx, song in enumerate(similar_songs[:settings.ui.max_similar_songs]):
            with cols[idx]:
                song_title_text = song.get('title', '')
                reason_text = song.get('reason', '')
                if st.button(
                    f"**{song_title_text}**\n\n_{reason_text}_",
                    key=f"similar_{idx}",
                    use_container_width=True
                ):
                    results = search_youtube(song_title_text)
                    if results:
                        st.session_state['selected_url'] = results[0]['url']
                        st.session_state['selected_title'] = results[0]['title']
                        if 'karaoke_data' in st.session_state:
                            del st.session_state['karaoke_data']
                        st.rerun()

    st.divider()
    song_title = st.session_state.get('selected_title', 'a song')
    segments = data.get('segments', [])

    col1, col2, col3 = st.columns(3)
    with col1:
        shareable_html = render_shareable_karaoke(
            song_title,
            data['audio_base64'],
            data['segments'],
            data['audio_format']
        )
        safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in song_title)
        st.download_button(
            "🎁 Share Karaoke Experience",
            shareable_html,
            file_name=f"{safe_filename}_karaoke.html",
            mime="text/html",
            use_container_width=True,
            help="Download an HTML file with the full karaoke player"
        )
    with col2:
        full_lyrics = f"🎶 {song_title}\n\n"
        for seg in segments:
            if seg.get('text'):
                full_lyrics += f"{seg['text']}\n"
                if seg.get('romanized'):
                    full_lyrics += f"({seg['romanized']})\n"
                if seg.get('translation'):
                    full_lyrics += f"→ {seg['translation']}\n"
                if seg.get('meaning'):
                    full_lyrics += f"💭 {seg['meaning']}\n"
                full_lyrics += "\n"
        st.download_button(
            "📝 Download Lyrics",
            full_lyrics,
            file_name=f"{safe_filename}_lyrics.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col3:
        if st.button("🎶 New Song", use_container_width=True):
            for key in ['selected_url', 'selected_title', 'karaoke_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

else:
    # -------------------------------------------------------------------------
    # No song playing: "You might also like", pipeline (progress bar is above tabs)
    # -------------------------------------------------------------------------
    # After song ends: show "You might also like"
    post_play_similar = st.session_state.get('post_play_similar_songs', [])
    post_play_title = st.session_state.get('post_play_title', '')
    if post_play_similar:
        st.divider()
        st.markdown("#### 🎵 You might also like")
        if post_play_title:
            st.caption(f"Just played: **{post_play_title}** — try one of these next or search above.")
        cols = st.columns(min(len(post_play_similar), settings.ui.max_similar_songs))
        for idx, song in enumerate(post_play_similar[:settings.ui.max_similar_songs]):
            with cols[idx]:
                song_title_text = song.get('title', '')
                reason_text = song.get('reason', '')
                if st.button(
                    f"**{song_title_text}**\n\n_{reason_text}_",
                    key=f"post_play_similar_{idx}",
                    use_container_width=True
                ):
                    results = search_youtube(song_title_text)
                    if results:
                        st.session_state['selected_url'] = results[0]['url']
                        st.session_state['selected_title'] = results[0]['title']
                        st.session_state['auto_process'] = True
                        for k in ['post_play_similar_songs', 'post_play_title']:
                            st.session_state.pop(k, None)
                        st.rerun()
    
    # -------------------------------------------------------------------------
    # Processing Pipeline
    # -------------------------------------------------------------------------
    if 'selected_url' in st.session_state and 'karaoke_data' not in st.session_state:
        # Clear "just played" suggestions when user picks a new song
        for k in ['post_play_similar_songs', 'post_play_title']:
            st.session_state.pop(k, None)
        # Check cache first
        cached = get_cached_result(st.session_state['selected_url'], "auto")
        if cached:
            st.session_state['karaoke_data'] = cached
            st.rerun()
        
        # Check rate limit (cached songs don't count)
        is_allowed, remaining = check_rate_limit()
        if not is_allowed:
            st.error("⏳ **Daily limit reached!** You've processed 10 songs today. Come back tomorrow!")
            st.info("💡 Tip: Songs you've already processed are cached and don't count toward the limit.")
            del st.session_state['selected_url']
            st.stop()
        
        if st.session_state.get('auto_process'):
            del st.session_state['auto_process']

            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    # Use progress widgets created above (above "Select a Song"); fallback if needed
                    if progress_bar is None:
                        progress_bar = st.progress(0)
                    if step_display is None:
                        step_display = st.empty()
                    if detail_display is None:
                        detail_display = st.empty()
                    if time_display is None:
                        time_display = st.empty()
                    start_time = time.time()
                    estimate_remaining_str = None  # set after download for steps 2–3
                    
                    steps = ["Download", "Transcribe", "Interpret"]
                    
                    def update_progress(step_num: int, detail: str = ""):
                        # Step 3 = interpret; bar only reaches 100% when player is ready (see Complete block)
                        p = (step_num / 3) if step_num < 3 else 2.5 / 3
                        progress_bar.progress(p)
                        step_text = "  →  ".join([
                            f"**{s}**" if i == step_num - 1 else f"~~{s}~~" if i < step_num - 1 else s
                            for i, s in enumerate(steps)
                        ])
                        step_display.markdown(f"Step {step_num}/3: {step_text}")
                        if detail:
                            detail_display.caption(detail)
                        elapsed = time.time() - start_time
                        if estimate_remaining_str and step_num >= 2:
                            time_display.caption(f"⏱️ {elapsed:.1f}s elapsed • Typically {estimate_remaining_str} remaining")
                        else:
                            time_display.caption(f"⏱️ {elapsed:.1f}s elapsed")
                    
                    # Step 1: Check duration and download
                    update_progress(1, "Checking video duration...")
                    
                    video_duration = get_video_duration(st.session_state['selected_url'])
                    if video_duration > settings.audio.max_duration_seconds:
                        mins = video_duration // 60
                        st.error(f"⏱️ This video is {mins} minutes long. Please choose a song under 10 minutes.")
                        progress_bar.empty()
                        step_display.empty()
                        detail_display.empty()
                        time_display.empty()
                        del st.session_state['selected_url']
                        st.stop()
                    
                    update_progress(1, "Fetching audio from YouTube...")
                    download_messages = [
                        "Connecting to YouTube...",
                        "Downloading audio stream...",
                        "Converting to MP3...",
                    ]
                    with animated_status(detail_display, download_messages):
                        audio_path = download_audio(st.session_state['selected_url'], tmp_dir)
                    
                    # Publish time estimate for transcription + interpretation (sets user expectation)
                    low_sec, high_sec = estimate_transcription_and_interpretation_sec(video_duration)
                    estimate_remaining_str = format_time_estimate(low_sec, high_sec)
                    
                    # Step 2: Transcribe
                    update_progress(2, "Using Whisper AI (auto-detecting language)...")
                    
                    skeleton_placeholder = st.empty()
                    show_skeleton_loading(skeleton_placeholder, 6)
                    
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
                    
                    text_segments = [s for s in segments if s['text'].strip()]
                    unique_count = len(set(s['text'].strip().lower() for s in text_segments))
                    detail_display.caption(f"✓ Found {len(text_segments)} lyric lines ({unique_count} unique)")
                    time.sleep(0.5)
                    
                    # Step 3: Interpret
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
                        interpreted_segments, detected_mood, song_summary = interpret_segments(segments)
                    
                    # Get similar songs
                    detail_display.caption("Finding similar songs...")
                    similar_songs = get_similar_songs(
                        st.session_state.get('selected_title', ''),
                        detected_mood,
                        song_summary
                    )
                    
                    # Build player
                    detail_display.caption("Building karaoke player...")
                    audio_base64 = get_audio_base64(audio_path)
                    
                    audio_ext = os.path.splitext(audio_path)[1].lstrip('.')
                    audio_format = 'webm' if audio_ext == 'webm' else 'mpeg'
                    
                    # Complete
                    skeleton_placeholder.empty()
                    progress_bar.progress(1.0)
                    total_time = time.time() - start_time
                    step_display.markdown("✅ **Ready to play!**")
                    detail_display.caption(f"Processed {len(text_segments)} lines in {total_time:.1f}s • Mood: {detected_mood}")
                    time_display.empty()
                    
                    # Store data
                    karaoke_data = {
                        'audio_base64': audio_base64,
                        'segments': interpreted_segments,
                        'audio_format': audio_format,
                        'mood': detected_mood,
                        'summary': song_summary,
                        'similar_songs': similar_songs
                    }
                    st.session_state['karaoke_data'] = karaoke_data
                    
                    # Save to cache
                    save_to_cache(
                        st.session_state['selected_url'],
                        "auto",
                        karaoke_data,
                        title=st.session_state.get('selected_title', 'Unknown')
                    )
                    
                    # Increment rate limit counter
                    increment_rate_limit()
                    
                    st.rerun()
                    
                except Exception as e:
                    logger.exception("Processing failed")
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Welcome message
    if 'selected_url' not in st.session_state:
        st.info("👆 Search for a song — works with 99+ languages!")
        
        st.markdown("### How it works")
        st.markdown("""
        1. **Search** for any song
        2. **AI transcribes** the lyrics (auto-detects language)
        3. **AI interprets** meaning with cultural context
        4. **Karaoke mode** syncs lyrics as the song plays
        
        **Works with any language:** French, Spanish, Arabic, Hindi, Korean, Japanese, Swahili, Portuguese, and many more!
        
        Perfect for:
        - 🌍 Understanding songs from any culture or language
        - 📚 Discovering idioms, metaphors, and cultural references  
        - 🎤 Singing along with phonetic pronunciations
        """)
        
        # Top foreign-language picks for smooth onboarding
        st.markdown("### 🎵 Try now")
        _french = CURATED_SONGS["🇫🇷 French"][0]
        _spanish = CURATED_SONGS["🇪🇸 Spanish"][0]
        _korean = CURATED_SONGS["🇰🇷 Korean"][0]
        _japanese = CURATED_SONGS["🇯🇵 Japanese"][0]
        onboarding_songs = [_french, _spanish, _korean, _japanese]
        cols = st.columns(min(4, len(onboarding_songs)))
        for idx, song in enumerate(onboarding_songs):
            with cols[idx]:
                if st.button(
                    f"▶️ **{song['title']}**\n{song['artist']}",
                    key=f"onboarding_{idx}",
                    use_container_width=True
                ):
                    with st.spinner("Finding best version..."):
                        results = search_youtube(song['query'], max_results=1)
                    if results:
                        st.session_state['selected_url'] = results[0]['url']
                        st.session_state['selected_title'] = f"{song['title']} - {song['artist']}"
                        st.session_state['auto_process'] = True
                        st.rerun()
                    else:
                        st.error("Could not find this song on YouTube")
