"""
YouTube-related utilities.
Handles video search, metadata extraction, and audio download.
Uses Cobalt API when configured (works on Streamlit Cloud); falls back to yt-dlp locally.
"""

import json
import logging
import os
import subprocess
import urllib.parse
import urllib.request
from typing import List, Optional, Dict, Any

from surasa.config.settings import settings

logger = logging.getLogger(__name__)

# User-Agent for Cobalt API (browser-like to avoid bot blocking on public instances)
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0"
}


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    
    Args:
        url: YouTube URL (youtube.com/watch?v=..., youtu.be/..., etc.)
        
    Returns:
        Video ID string or None if not found.
    """
    if not url:
        return None
        
    # Handle youtube.com/watch?v=VIDEO_ID
    if "youtube.com" in url and "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
        return video_id if video_id else None
    
    # Handle youtu.be/VIDEO_ID
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return video_id if video_id else None
    
    return None


def get_thumbnail_url(url_or_video_id: str, quality: str = "mqdefault") -> str:
    """
    Get YouTube thumbnail URL.
    
    Args:
        url_or_video_id: Either a YouTube URL or video ID.
        quality: Thumbnail quality (default, mqdefault, hqdefault, sddefault, maxresdefault)
        
    Returns:
        Thumbnail URL string.
    """
    video_id = url_or_video_id
    
    # If it looks like a URL, extract the video ID
    if "youtube.com" in url_or_video_id or "youtu.be" in url_or_video_id:
        video_id = extract_video_id(url_or_video_id)
    
    if not video_id:
        return ""
    
    return f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"


def score_video_relevance(video: Dict[str, Any], query: str) -> int:
    """
    Score a video's relevance for finding the original song.
    Higher score = better match.
    
    Args:
        video: Video metadata dict with 'title', 'channel', 'duration'.
        query: Original search query.
        
    Returns:
        Integer score (can be negative for poor matches).
    """
    score = 0
    title_lower = video.get('title', '').lower()
    channel_lower = video.get('channel', '').lower()
    
    # Prefer official content
    if 'official' in title_lower:
        score += 50
    if 'official' in channel_lower or 'vevo' in channel_lower:
        score += 40
    
    # Prefer music videos over others
    if 'official video' in title_lower or 'official mv' in title_lower:
        score += 30
    if 'official audio' in title_lower:
        score += 25
    if 'lyric' in title_lower and 'video' in title_lower:
        score += 20
    
    # Penalize covers, remixes, live versions
    penalties = [
        ('cover', -100),
        ('remix', -80),
        ('karaoke', -100),
        ('instrumental', -100),
        ('slowed', -80),
        ('reverb', -80),
        ('reaction', -100),
    ]
    for keyword, penalty in penalties:
        if keyword in title_lower:
            score += penalty
    
    # Penalize live versions (unless official)
    if 'live' in title_lower and 'official' not in title_lower:
        score -= 50
    
    # Prefer reasonable duration (2-7 minutes typical for songs)
    duration = video.get('duration_sec', 0)
    if isinstance(duration, int) and 120 <= duration <= 420:
        score += 10
    
    return score


def search_youtube(query: str, max_results: int = None) -> List[Dict[str, Any]]:
    """
    Search YouTube and return ranked results.
    
    Args:
        query: Search query string.
        max_results: Maximum results to return (default from settings).
        
    Returns:
        List of video dicts with title, url, channel, duration, thumbnail, score.
    """
    if max_results is None:
        max_results = settings.ui.max_search_results
    
    try:
        # Search for more results to filter and rank
        result = subprocess.run(
            ["yt-dlp", f"ytsearch10:{query}", "--dump-json", "--flat-playlist"],
            capture_output=True, 
            text=True, 
            timeout=settings.api.request_timeout
        )
        
        raw_results = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                video_id = data.get('id', '')
                video = {
                    'title': data.get('title', 'Unknown'),
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'channel': data.get('channel', data.get('uploader', 'Unknown')),
                    'duration': data.get('duration_string', ''),
                    'duration_sec': data.get('duration', 0),
                    'view_count': data.get('view_count', 0),
                    'thumbnail': get_thumbnail_url(video_id),
                }
                video['score'] = score_video_relevance(video, query)
                raw_results.append(video)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse search result: {line[:100]}")
                continue
        
        # Filter out videos longer than max duration
        max_duration = settings.audio.max_duration_seconds
        raw_results = [v for v in raw_results if v.get('duration_sec', 0) <= max_duration]
        
        # Sort by score (highest first), then by view count
        raw_results.sort(key=lambda x: (x['score'], x.get('view_count', 0)), reverse=True)
        
        return raw_results[:max_results]
        
    except subprocess.TimeoutExpired:
        logger.error("YouTube search timed out")
        return []
    except Exception as e:
        logger.error(f"YouTube search failed: {e}")
        return []


def get_video_duration(url: str) -> int:
    """
    Get video duration in seconds.
    
    Args:
        url: YouTube video URL.
        
    Returns:
        Duration in seconds, or 0 if unable to fetch.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True, 
            text=True, 
            timeout=15
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get('duration', 0)
    except subprocess.TimeoutExpired:
        logger.warning(f"Duration check timed out for: {url}")
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse duration response for: {url}")
    except Exception as e:
        logger.warning(f"Failed to get duration: {e}")
    
    return 0


def _download_audio_via_cobalt(youtube_url: str, output_dir: str) -> Optional[str]:
    """
    Download audio using Cobalt API (works on Streamlit Cloud).
    
    Returns:
        Path to the downloaded file, or None if Cobalt is not configured or request fails.
    """
    base = (settings.audio.cobalt_api_url or "").strip()
    if not base:
        return None
    
    api_url = base.rstrip("/") + "/"
    
    body = {
        "url": youtube_url,
        "downloadMode": "audio",
        "audioFormat": "mp3",
        "filenameStyle": "basic",
    }
    
    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **_REQUEST_HEADERS,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        status = data.get("status")
        if status not in ("tunnel", "redirect"):
            if status == "error":
                err = data.get("error", {})
                logger.warning(f"Cobalt API error: {err.get('code', 'unknown')}")
            return None
        
        audio_url = data.get("url")
        filename = data.get("filename", "audio.mp3")
        if not audio_url:
            return None
        
        # Sanitize filename for local save
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        if not safe_name.lower().endswith(".mp3"):
            safe_name = "audio.mp3"
        output_path = os.path.join(output_dir, safe_name)
        
        download_req = urllib.request.Request(audio_url, headers=_REQUEST_HEADERS)
        with urllib.request.urlopen(download_req, timeout=settings.audio.download_timeout) as resp:
            with open(output_path, "wb") as f:
                f.write(resp.read())
        
        logger.info(f"Cobalt: saved audio to {output_path}")
        return output_path
        
    except urllib.error.HTTPError as e:
        logger.warning(f"Cobalt API HTTP error: {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.warning(f"Cobalt API URL error: {e.reason}")
        return None
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning(f"Cobalt API error: {e}")
        return None


def _download_audio_via_ytdlp(url: str, output_dir: str) -> str:
    """Download audio using yt-dlp (for local use). Raises on failure."""
    output_template = os.path.join(output_dir, "audio.%(ext)s")
    
    result = subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", settings.audio.audio_quality,
            "-o", output_template,
            "--no-playlist",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=settings.audio.download_timeout,
    )
    
    for f in os.listdir(output_dir):
        if f.startswith("audio."):
            return os.path.join(output_dir, f)
    
    raise Exception(f"Download failed: {result.stderr}")


def download_audio(url: str, output_dir: str) -> str:
    """
    Download audio from YouTube URL.
    Tries Cobalt API first (works on Streamlit Cloud), then yt-dlp.
    
    Args:
        url: YouTube video URL.
        output_dir: Directory to save the audio file.
        
    Returns:
        Path to the downloaded audio file.
        
    Raises:
        Exception: If both methods fail.
    """
    path = _download_audio_via_cobalt(url, output_dir)
    if path:
        return path
    
    try:
        return _download_audio_via_ytdlp(url, output_dir)
    except Exception as e:
        raise Exception(
            f"Download failed: Cobalt API unavailable and yt-dlp failed. {e}"
        )


def get_youtube_suggestions(query: str) -> List[str]:
    """
    Get autocomplete suggestions from YouTube.
    
    Args:
        query: Partial search query.
        
    Returns:
        List of suggestion strings.
    """
    if not query or len(query) < 2:
        return []
    
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://suggestqueries.google.com/complete/search?client=youtube&ds=yt&q={encoded_query}"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            data = response.read().decode('utf-8')
            # Parse the JSONP response
            start = data.find('[[')
            end = data.rfind(']]') + 2
            if start > 0 and end > start:
                suggestions_data = json.loads(data[start:end])
                return [s[0] for s in suggestions_data if s][:8]
    except Exception as e:
        logger.debug(f"Failed to get suggestions: {e}")
    
    return []
