"""Reusable UI components."""

import json
from typing import List


def render_animated_status(messages: List[str], interval_ms: int = 2000) -> str:
    """
    Create HTML/JS that animates through status messages client-side.
    
    Args:
        messages: List of status message strings.
        interval_ms: Interval between messages in milliseconds.
        
    Returns:
        HTML string with embedded JavaScript.
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


def render_skeleton_loading(num_lines: int = 5) -> str:
    """
    Render skeleton loading placeholder.
    
    Args:
        num_lines: Number of skeleton lines to show.
        
    Returns:
        HTML string for skeleton loading.
    """
    return """
    <style>
    @keyframes subtle-pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.5; }
    }
    .skeleton-line-subtle {
        background: rgba(128, 128, 128, 0.15);
        border-radius: 4px;
        height: 12px;
        margin: 10px 0;
        animation: subtle-pulse 2s ease-in-out infinite;
    }
    .skeleton-line-subtle.short { width: 60%; }
    .skeleton-line-subtle.medium { width: 75%; }
    .skeleton-line-subtle.long { width: 85%; }
    .skeleton-wrapper {
        padding: 20px 0;
        opacity: 0.7;
    }
    </style>
    <div class="skeleton-wrapper">
        <div class="skeleton-line-subtle long"></div>
        <div class="skeleton-line-subtle medium" style="animation-delay: 0.2s"></div>
        <div class="skeleton-line-subtle short" style="animation-delay: 0.4s"></div>
        <div class="skeleton-line-subtle long" style="animation-delay: 0.6s"></div>
        <div class="skeleton-line-subtle medium" style="animation-delay: 0.8s"></div>
    </div>
    """
