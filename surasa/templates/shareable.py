"""
Shareable standalone HTML karaoke file template.
"""

import json
from typing import List, Dict, Any


def render_shareable_karaoke(
    title: str,
    audio_base64: str,
    segments: List[Dict[str, Any]],
    audio_format: str = "mp3"
) -> str:
    """
    Create a standalone HTML file with the full karaoke experience.
    
    Args:
        title: Song title.
        audio_base64: Base64 encoded audio data.
        segments: List of lyric segments with interpretations.
        audio_format: Audio format (mp3, webm, etc.).
        
    Returns:
        Complete standalone HTML document string.
    """
    segments_json = json.dumps(segments)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Surasa Karaoke</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }}
        
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 1.8em;
            margin-bottom: 8px;
        }}
        
        .header .subtitle {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .header .brand {{
            color: #00d4ff;
            font-size: 0.85em;
            margin-top: 10px;
        }}
        
        .karaoke-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            max-height: 80vh;
            position: relative;
        }}
        
        .controls-section {{
            position: sticky;
            top: 0;
            z-index: 10;
            background: rgba(26, 26, 46, 0.95);
            padding: 20px 24px;
            border-radius: 16px 16px 0 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .seek-container {{
            margin: 10px 0;
        }}
        
        .seek-bar {{
            width: 100%;
            height: 8px;
            -webkit-appearance: none;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            cursor: pointer;
        }}
        
        .seek-bar::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        .time-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
        }}
        
        .time-current {{ color: #00d4ff; font-weight: 500; }}
        .time-duration {{ color: #888; }}
        
        .playback-controls {{
            display: flex;
            gap: 10px;
        }}
        
        .skip-btn {{
            background: rgba(255,255,255,0.1);
            border: none;
            color: white;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        
        .skip-btn:hover {{
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
        }}
        
        .lyrics-container {{
            flex: 1;
            overflow-y: auto;
            scroll-behavior: smooth;
            padding: 20px 24px;
            background: rgba(0,0,0,0.2);
            border-radius: 0 0 16px 16px;
            scrollbar-width: thin;
            scrollbar-color: rgba(255,255,255,0.2) transparent;
        }}
        
        .lyrics-container::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .lyrics-container::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        .lyrics-container::-webkit-scrollbar-thumb {{
            background: rgba(255,255,255,0.2);
            border-radius: 3px;
        }}
        
        .lyric-segment {{
            padding: 16px;
            margin: 10px 0;
            border-radius: 10px;
            transition: all 0.3s ease;
            opacity: 0.4;
            border-left: 3px solid transparent;
            cursor: pointer;
        }}
        
        .lyric-segment:hover {{
            opacity: 0.7;
            background: rgba(255,255,255,0.03);
        }}
        
        .lyric-segment.active {{
            opacity: 1;
            background: rgba(0, 212, 255, 0.1);
            border-left: 3px solid #00d4ff;
            transform: scale(1.02);
        }}
        
        .lyric-segment.past {{
            opacity: 0.6;
        }}
        
        .original {{
            font-size: 1.4em;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        
        .romanized {{
            font-size: 1.1em;
            color: #ffd700;
            margin-bottom: 8px;
            font-style: italic;
        }}
        
        .translation {{
            font-size: 1.15em;
            color: #00d4ff;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .meaning {{
            font-size: 0.9em;
            color: #aaa;
            padding: 10px 14px;
            background: rgba(0, 212, 255, 0.08);
            border-radius: 8px;
            line-height: 1.5;
        }}
        
        .time-badge {{
            font-size: 0.75em;
            color: #666;
            margin-bottom: 6px;
        }}
        
        .celebration-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.5s ease;
            z-index: 100;
            border-radius: 16px;
            overflow: hidden;
        }}
        
        .celebration-overlay.active {{
            opacity: 1;
        }}
        
        .celebration-glow {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 150%;
            height: 150%;
            background: radial-gradient(circle, #00d4ff 0%, transparent 70%);
            opacity: 0.3;
            animation: pulse-glow 2s ease-in-out infinite;
        }}
        
        @keyframes pulse-glow {{
            0%, 100% {{ transform: translate(-50%, -50%) scale(1); opacity: 0.3; }}
            50% {{ transform: translate(-50%, -50%) scale(1.1); opacity: 0.5; }}
        }}
        
        .sparkle {{
            position: absolute;
            width: 10px;
            height: 10px;
            background: #00d4ff;
            border-radius: 50%;
            animation: sparkle-float 3s ease-out forwards;
        }}
        
        @keyframes sparkle-float {{
            0% {{ opacity: 1; transform: translateY(0) scale(1); }}
            100% {{ opacity: 0; transform: translateY(-100px) scale(0); }}
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.85em;
        }}
        
        .footer a {{
            color: #00d4ff;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎤 {title}</h1>
            <p class="subtitle">Click any line to jump to that moment</p>
            <p class="brand">✨ Created with Surasa - The Song Meaning App</p>
        </div>
        
        <div class="karaoke-container" style="position: relative;">
            <div class="celebration-overlay" id="celebrationOverlay">
                <div class="celebration-glow"></div>
            </div>
            
            <div class="controls-section">
                <audio id="audioPlayer" style="display:none;">
                    <source src="data:audio/{audio_format};base64,{audio_base64}" type="audio/{audio_format}">
                </audio>
                
                <div class="seek-container">
                    <input type="range" id="seekBar" class="seek-bar" value="0" min="0" max="100" step="0.1">
                    <div class="time-row">
                        <span class="time-current" id="currentTime">0:00</span>
                        <div class="playback-controls">
                            <button class="skip-btn" id="skipBack">⏪ 10s</button>
                            <button class="skip-btn" id="playPause">▶️ Play</button>
                            <button class="skip-btn" id="skipForward">10s ⏩</button>
                        </div>
                        <span class="time-duration" id="duration">0:00</span>
                    </div>
                </div>
            </div>
            
            <div class="lyrics-container" id="lyricsContainer"></div>
        </div>
        
        <div class="footer">
            <p>Lyrics interpreted by AI • Share the joy of music! 🎶</p>
        </div>
    </div>
    
    <script>
        const segments = {segments_json};
        const audio = document.getElementById('audioPlayer');
        const container = document.getElementById('lyricsContainer');
        const seekBar = document.getElementById('seekBar');
        const currentTimeDisplay = document.getElementById('currentTime');
        const durationDisplay = document.getElementById('duration');
        const playPauseBtn = document.getElementById('playPause');
        
        // Utility function - must be defined before use
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
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
                        <div class="translation">${{translation}}</div>
                        ${{meaning ? `<div class="meaning">${{meaning}}</div>` : ''}}
                    </div>
                `;
            }}
        }});
        container.innerHTML = lyricsHTML;
        
        function updatePlayPauseBtn() {{
            playPauseBtn.textContent = audio.paused ? '▶️ Play' : '⏸️ Pause';
        }}
        
        playPauseBtn.addEventListener('click', () => {{
            if (audio.paused) audio.play();
            else audio.pause();
            updatePlayPauseBtn();
        }});
        
        audio.addEventListener('play', updatePlayPauseBtn);
        audio.addEventListener('pause', updatePlayPauseBtn);
        
        document.getElementById('skipBack').addEventListener('click', () => {{
            audio.currentTime = Math.max(0, audio.currentTime - 10);
        }});
        
        document.getElementById('skipForward').addEventListener('click', () => {{
            audio.currentTime = Math.min(audio.duration, audio.currentTime + 10);
        }});
        
        let isSeeking = false;
        seekBar.addEventListener('mousedown', () => isSeeking = true);
        seekBar.addEventListener('mouseup', () => isSeeking = false);
        seekBar.addEventListener('input', () => {{
            audio.currentTime = (seekBar.value / 100) * audio.duration;
        }});
        
        audio.addEventListener('loadedmetadata', () => {{
            durationDisplay.textContent = formatTime(audio.duration);
        }});
        
        audio.addEventListener('timeupdate', () => {{
            const currentTime = audio.currentTime;
            const duration = audio.duration || 0;
            
            if (!isSeeking && duration > 0) {{
                seekBar.value = (currentTime / duration) * 100;
            }}
            
            currentTimeDisplay.textContent = formatTime(currentTime);
            if (duration > 0) durationDisplay.textContent = formatTime(duration);
            
            segments.forEach((seg, idx) => {{
                const el = document.getElementById(`segment-${{idx}}`);
                if (!el) return;
                
                if (currentTime >= seg.start && currentTime < seg.end) {{
                    el.classList.add('active');
                    el.classList.remove('past');
                    el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }} else if (currentTime >= seg.end) {{
                    el.classList.remove('active');
                    el.classList.add('past');
                }} else {{
                    el.classList.remove('active', 'past');
                }}
            }});
        }});
        
        document.querySelectorAll('.lyric-segment').forEach(el => {{
            el.addEventListener('click', () => {{
                audio.currentTime = parseFloat(el.dataset.start);
                audio.play();
                updatePlayPauseBtn();
            }});
        }});
        
        // Celebration
        const celebrationOverlay = document.getElementById('celebrationOverlay');
        
        function createSparkles() {{
            const colors = ['#00d4ff', '#FFD700', '#FF6B9D', '#6BFFB8'];
            for (let i = 0; i < 20; i++) {{
                const sparkle = document.createElement('div');
                sparkle.className = 'sparkle';
                sparkle.style.left = Math.random() * 100 + '%';
                sparkle.style.top = Math.random() * 100 + '%';
                sparkle.style.background = colors[Math.floor(Math.random() * colors.length)];
                sparkle.style.animationDelay = Math.random() * 2 + 's';
                sparkle.style.width = (Math.random() * 8 + 4) + 'px';
                sparkle.style.height = sparkle.style.width;
                celebrationOverlay.appendChild(sparkle);
            }}
        }}
        
        function showCelebration() {{
            createSparkles();
            celebrationOverlay.classList.add('active');
            setTimeout(() => {{
                celebrationOverlay.classList.remove('active');
                celebrationOverlay.querySelectorAll('.sparkle').forEach(s => s.remove());
            }}, 4000);
        }}
        
        audio.addEventListener('ended', () => {{
            updatePlayPauseBtn();
            showCelebration();
        }});
    </script>
</body>
</html>"""
