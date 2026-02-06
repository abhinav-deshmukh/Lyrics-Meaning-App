"""Utility functions for Surasa."""

from .youtube import (
    extract_video_id,
    get_thumbnail_url,
    search_youtube,
    get_video_duration,
    download_audio,
    get_youtube_suggestions,
    score_video_relevance,
)
from .cache import (
    get_cache_key,
    get_cached_result,
    save_to_cache,
    get_cached_songs,
    clear_cache,
)

__all__ = [
    # YouTube utilities
    "extract_video_id",
    "get_thumbnail_url",
    "search_youtube",
    "get_video_duration",
    "download_audio",
    "get_youtube_suggestions",
    "score_video_relevance",
    # Cache utilities
    "get_cache_key",
    "get_cached_result",
    "save_to_cache",
    "get_cached_songs",
    "clear_cache",
]
