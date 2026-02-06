"""
Caching utilities for processed songs.
Provides file-based caching with metadata for history feature.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from surasa.config.settings import settings
from surasa.utils.youtube import extract_video_id, get_thumbnail_url

logger = logging.getLogger(__name__)

# Ensure cache directory exists
os.makedirs(settings.cache.cache_dir, exist_ok=True)


def get_cache_key(url: str, language: str = "auto") -> str:
    """
    Generate cache key from URL and language.
    
    Args:
        url: YouTube URL.
        language: Language code or "auto".
        
    Returns:
        MD5 hash string.
    """
    return hashlib.md5(f"{url}:{language}".encode()).hexdigest()


def get_cached_result(url: str, language: str = "auto") -> Optional[Dict[str, Any]]:
    """
    Try to get cached result for a song.
    
    Args:
        url: YouTube URL.
        language: Language code or "auto".
        
    Returns:
        Cached data dict or None if not found.
    """
    cache_key = get_cache_key(url, language)
    cache_file = os.path.join(settings.cache.cache_dir, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupt cache file {cache_key}: {e}")
        return None
    except IOError as e:
        logger.warning(f"Failed to read cache file {cache_key}: {e}")
        return None


def save_to_cache(
    url: str, 
    language: str, 
    data: Dict[str, Any], 
    title: Optional[str] = None
) -> bool:
    """
    Save processed song to cache with metadata.
    
    Args:
        url: YouTube URL.
        language: Language code or "auto".
        data: Song data to cache.
        title: Song title for metadata.
        
    Returns:
        True if saved successfully, False otherwise.
    """
    cache_key = get_cache_key(url, language)
    cache_file = os.path.join(settings.cache.cache_dir, f"{cache_key}.json")
    
    # Extract video ID for thumbnail
    video_id = extract_video_id(url)
    thumbnail = get_thumbnail_url(video_id) if video_id else ""
    
    # Add metadata for history feature
    data['_meta'] = {
        'url': url,
        'title': title or 'Unknown',
        'thumbnail': thumbnail,
        'cached_at': time.strftime('%Y-%m-%d %H:%M')
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return True
    except IOError as e:
        logger.error(f"Failed to write cache file {cache_key}: {e}")
        return False


def get_cached_songs() -> List[Dict[str, Any]]:
    """
    Get list of all cached songs for history feature.
    
    Returns:
        List of song metadata dicts, sorted by most recent.
    """
    songs = []
    
    try:
        for filename in os.listdir(settings.cache.cache_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(settings.cache.cache_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    meta = data.get('_meta', {})
                    
                    if not meta:
                        continue
                    
                    # Generate thumbnail from URL if not stored
                    thumbnail = meta.get('thumbnail', '')
                    if not thumbnail and meta.get('url'):
                        video_id = extract_video_id(meta['url'])
                        if video_id:
                            thumbnail = get_thumbnail_url(video_id)
                    
                    songs.append({
                        'title': meta.get('title', 'Unknown'),
                        'url': meta.get('url', ''),
                        'thumbnail': thumbnail,
                        'cached_at': meta.get('cached_at', ''),
                        'cache_key': filename.replace('.json', '')
                    })
                    
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Skipping corrupt cache file {filename}: {e}")
                continue
                
    except IOError as e:
        logger.error(f"Failed to list cache directory: {e}")
    
    # Sort by most recent
    songs.sort(key=lambda x: x.get('cached_at', ''), reverse=True)
    return songs


def clear_cache() -> int:
    """
    Clear all cached songs.
    
    Returns:
        Number of files deleted.
    """
    deleted = 0
    try:
        for filename in os.listdir(settings.cache.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(settings.cache.cache_dir, filename)
                try:
                    os.remove(filepath)
                    deleted += 1
                except IOError as e:
                    logger.warning(f"Failed to delete {filename}: {e}")
    except IOError as e:
        logger.error(f"Failed to list cache directory: {e}")
    
    return deleted
