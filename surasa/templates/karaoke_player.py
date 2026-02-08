"""
Karaoke player HTML/CSS/JS template rendering.
"""

import html
import json
import os
from typing import List, Dict, Any, Optional

from surasa.config.settings import MOOD_THEMES


def _load_css() -> str:
    """Load the karaoke CSS from file."""
    css_path = os.path.join(os.path.dirname(__file__), 'styles', 'karaoke.css')
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError:
        # Fallback to minimal inline styles if file not found
        return """
        .karaoke-container { 
            font-family: sans-serif; 
            background: #1a1a2e; 
            padding: 20px; 
            border-radius: 12px; 
        }
        """


def _get_karaoke_javascript() -> str:
    """Return the karaoke player JavaScript."""
    return """
    <script>
        const segments = SEGMENTS_JSON;
        const audio = document.getElementById('audioPlayer');
        const container = document.getElementById('lyricsContainer');
        const seekBar = document.getElementById('seekBar');
        const currentTimeDisplay = document.getElementById('currentTime');
        const durationDisplay = document.getElementById('duration');
        const playPauseBtn = document.getElementById('playPauseBtn');
        
        // Utility function - must be defined before use
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
        
        // Build lyrics HTML
        let lyricsHTML = '';
        segments.forEach((seg, idx) => {
            if (seg.text) {
                const romanized = seg.romanized || '';
                const translation = seg.translation || '';
                const meaning = seg.meaning || '';
                lyricsHTML += `
                    <div class="lyric-segment" id="segment-${idx}" data-start="${seg.start}" data-end="${seg.end}">
                        <div class="time-badge">${formatTime(seg.start)}</div>
                        <div class="original">${seg.text}</div>
                        ${romanized ? `<div class="romanized">${romanized}</div>` : ''}
                        <div class="translation">${translation}</div>
                        ${meaning ? `<div class="meaning">${meaning}</div>` : ''}
                    </div>
                `;
            }
        });
        container.innerHTML = lyricsHTML;
        
        function updatePlayPauseBtn() {
            playPauseBtn.textContent = audio.paused ? '▶ Play' : '⏸ Pause';
        }
        
        playPauseBtn.addEventListener('click', () => {
            if (audio.paused) audio.play();
            else audio.pause();
            updatePlayPauseBtn();
        });
        
        audio.addEventListener('play', updatePlayPauseBtn);
        audio.addEventListener('pause', updatePlayPauseBtn);
        
        document.getElementById('skipBack').addEventListener('click', () => {
            audio.currentTime = Math.max(0, audio.currentTime - 10);
        });
        
        document.getElementById('skipForward').addEventListener('click', () => {
            audio.currentTime = Math.min(audio.duration, audio.currentTime + 10);
        });
        
        let isSeeking = false;
        seekBar.addEventListener('mousedown', () => isSeeking = true);
        seekBar.addEventListener('touchstart', () => isSeeking = true);
        seekBar.addEventListener('mouseup', () => isSeeking = false);
        seekBar.addEventListener('touchend', () => isSeeking = false);
        seekBar.addEventListener('input', () => {
            const time = (seekBar.value / 100) * audio.duration;
            audio.currentTime = time;
        });
        
        audio.addEventListener('loadedmetadata', () => {
            durationDisplay.textContent = formatTime(audio.duration);
        });
        
        // Sync lyrics with audio
        audio.addEventListener('timeupdate', () => {
            const currentTime = audio.currentTime;
            const duration = audio.duration || 0;
            
            // Update seek bar
            if (!isSeeking && duration > 0) {
                seekBar.value = (currentTime / duration) * 100;
            }
            
            currentTimeDisplay.textContent = formatTime(currentTime);
            if (duration > 0) durationDisplay.textContent = formatTime(duration);
            
            // Update active lyrics
            segments.forEach((seg, idx) => {
                const el = document.getElementById(`segment-${idx}`);
                if (!el) return;
                
                if (currentTime >= seg.start && currentTime < seg.end) {
                    if (!el.classList.contains('active')) {
                        el.classList.add('active');
                        el.classList.remove('past');
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                } else if (currentTime >= seg.end) {
                    el.classList.remove('active');
                    el.classList.add('past');
                } else {
                    el.classList.remove('active', 'past');
                }
            });
        });
        
        // Click lyrics to seek
        document.querySelectorAll('.lyric-segment').forEach(el => {
            el.addEventListener('click', () => {
                const start = parseFloat(el.dataset.start);
                audio.currentTime = start;
                audio.play();
                updatePlayPauseBtn();
            });
        });
        
        // Focus mode toggle
        const focusToggle = document.getElementById('focusToggle');
        const karaokeContainer = document.getElementById('karaokeContainer');
        
        focusToggle.addEventListener('change', () => {
            if (focusToggle.checked) {
                karaokeContainer.classList.add('focus-mode');
            } else {
                karaokeContainer.classList.remove('focus-mode');
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' && e.target.type !== 'checkbox') return;
            
            if (e.code === 'Space') {
                e.preventDefault();
                if (audio.paused) audio.play();
                else audio.pause();
                updatePlayPauseBtn();
            } else if (e.code === 'ArrowLeft') {
                audio.currentTime = Math.max(0, audio.currentTime - 5);
            } else if (e.code === 'ArrowRight') {
                audio.currentTime = Math.min(audio.duration, audio.currentTime + 5);
            } else if (e.code === 'KeyF') {
                focusToggle.checked = !focusToggle.checked;
                focusToggle.dispatchEvent(new Event('change'));
            }
        });
        
        // Celebration when song ends
        const celebrationOverlay = document.getElementById('celebrationOverlay');
        
        function createSparkles() {
            const colors = ['var(--accent-color)', '#FFD700', '#FF6B9D', '#6BFFB8'];
            for (let i = 0; i < 20; i++) {
                const sparkle = document.createElement('div');
                sparkle.className = 'sparkle';
                sparkle.style.left = Math.random() * 100 + '%';
                sparkle.style.top = Math.random() * 100 + '%';
                sparkle.style.background = colors[Math.floor(Math.random() * colors.length)];
                sparkle.style.animationDelay = Math.random() * 2 + 's';
                sparkle.style.width = (Math.random() * 8 + 4) + 'px';
                sparkle.style.height = sparkle.style.width;
                celebrationOverlay.appendChild(sparkle);
            }
        }
        
        function showCelebration() {
            createSparkles();
            celebrationOverlay.classList.add('active');
            
            // Auto-close after 4 seconds
            setTimeout(() => {
                celebrationOverlay.classList.remove('active');
                celebrationOverlay.querySelectorAll('.sparkle').forEach(s => s.remove());
            }, 4000);
        }
        
        // Dismiss celebration on click
        celebrationOverlay.addEventListener('click', () => {
            celebrationOverlay.classList.remove('active');
            celebrationOverlay.querySelectorAll('.sparkle').forEach(s => s.remove());
        });
        
        audio.addEventListener('ended', () => {
            updatePlayPauseBtn();
            showCelebration();
            // Auto-return to search after a short delay so user sees the search bar without clicking
            setTimeout(() => {
                const url = (window.top.location.origin || '') + (window.top.location.pathname || '/') + '?new_search=1';
                window.top.location.assign(url);
            }, 2500);
        });
        
        // Fullscreen functionality
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const playerWrapper = document.getElementById('playerWrapper');
        
        function updateFullscreenBtn() {
            const isFullscreen = document.fullscreenElement || document.webkitFullscreenElement;
            fullscreenBtn.textContent = isFullscreen ? '✕' : '⛶';
            fullscreenBtn.title = isFullscreen ? 'Exit Fullscreen' : 'Fullscreen';
        }
        
        fullscreenBtn.addEventListener('click', () => {
            if (document.fullscreenElement || document.webkitFullscreenElement) {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                }
            } else {
                if (playerWrapper.requestFullscreen) {
                    playerWrapper.requestFullscreen();
                } else if (playerWrapper.webkitRequestFullscreen) {
                    playerWrapper.webkitRequestFullscreen();
                }
            }
        });
        
        document.addEventListener('fullscreenchange', updateFullscreenBtn);
        document.addEventListener('webkitfullscreenchange', updateFullscreenBtn);
    </script>
    """


def render_karaoke_player(
    audio_base64: str,
    segments: List[Dict[str, Any]],
    audio_format: str = "mp3",
    mood: str = "playful",
    summary: str = "",
    similar_songs: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Render the karaoke player HTML with all CSS and JavaScript.
    
    Args:
        audio_base64: Base64 encoded audio data.
        segments: List of lyric segments with interpretations.
        audio_format: Audio format (mp3, webm, etc.).
        mood: Song mood for theming.
        summary: Song summary text.
        similar_songs: List of similar song suggestions.
        
    Returns:
        Complete HTML string for the karaoke player.
    """
    if similar_songs is None:
        similar_songs = []
    
    # Get theme colors
    theme = MOOD_THEMES.get(mood, MOOD_THEMES["playful"])
    accent = theme["accent"]
    bg1 = theme["bg1"]
    bg2 = theme["bg2"]
    
    # Escape summary
    summary_escaped = html.escape(summary) if summary else ""
    
    # Build components
    segments_json = json.dumps(segments)
    
    # Get CSS (with CSS variables set)
    base_css = _load_css()
    
    # Build the JavaScript with segments data
    js = _get_karaoke_javascript().replace('SEGMENTS_JSON', segments_json)
    
    return f"""
    <style>
        :root {{
            --accent-color: {accent};
            --bg-gradient-1: {bg1};
            --bg-gradient-2: {bg2};
            --spacing-xs: 8px;
            --spacing-sm: 12px;
            --spacing-md: 16px;
            --spacing-lg: 20px;
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
        }}
        {base_css}
    </style>
    
    <!-- Fullscreen wrapper -->
    <div class="player-wrapper" id="playerWrapper">
        <!-- About this song (above play track; only when summary is available) -->
        {f'''<div class="summary-card">
            <div class="summary-label">About this song</div>
            <div class="summary-text">{summary_escaped}</div>
        </div>''' if summary_escaped else ''}
        
        <!-- Player Controls (progress bar) -->
        <div class="audio-controls-standalone">
            <audio id="audioPlayer" style="display:none;">
                <source src="data:audio/{audio_format};base64,{audio_base64}" type="audio/{audio_format}">
            </audio>
            
            <div class="seek-container">
                <input type="range" id="seekBar" class="seek-bar" value="0" min="0" max="100" step="0.1">
                <div class="time-row">
                    <span class="time-current" id="currentTime">0:00</span>
                    <div class="playback-controls">
                        <button class="skip-btn" id="skipBack">-10s</button>
                        <button class="skip-btn play-btn" id="playPauseBtn">▶ Play</button>
                        <button class="skip-btn" id="skipForward">+10s</button>
                        <button class="skip-btn fullscreen-btn" id="fullscreenBtn" title="Fullscreen">⛶</button>
                    </div>
                    <span class="time-duration" id="duration">0:00</span>
                </div>
            </div>
        </div>
        
        <div class="karaoke-container" id="karaokeContainer" style="position: relative;">
            <!-- Celebration overlay (visual only, no text) -->
            <div class="celebration-overlay" id="celebrationOverlay">
                <div class="celebration-glow"></div>
            </div>
            
            <!-- Lyrics header -->
            <div class="lyrics-header">
                <span class="mood-badge">🎭 {mood}</span>
                <label class="focus-toggle">
                    <input type="checkbox" id="focusToggle"> Focus mode
                </label>
            </div>
            
            <!-- Lyrics Card -->
            <div class="lyrics-container" id="lyricsContainer"></div>
        </div>
    </div>
    
    {js}
    """
