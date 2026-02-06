"""
Audio transcription service using OpenAI Whisper.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI

from surasa.config.settings import settings

logger = logging.getLogger(__name__)


def transcribe_with_timestamps(
    audio_path: str, 
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Transcribe audio with segment-level timestamps.
    
    Args:
        audio_path: Path to audio file.
        language: Optional language hint (ISO code or "auto").
        
    Returns:
        List of segment dicts with 'start', 'end', 'text' keys.
        
    Raises:
        Exception: If transcription fails after retries.
    """
    client = OpenAI()
    max_retries = settings.api.max_retries
    
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as audio_file:
                params = {
                    "model": settings.api.whisper_model,
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                
                # Add language hint if specified (not "auto")
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
            
            logger.info(f"Transcribed {len(segments)} segments from {audio_path}")
            return segments
            
        except Exception as e:
            logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Transcription failed after {max_retries} attempts")
                raise
