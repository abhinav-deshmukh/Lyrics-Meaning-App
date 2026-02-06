"""
Lyrics interpretation service using Claude.
Provides translation, romanization, and cultural meaning.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional

from anthropic import Anthropic

from surasa.config.settings import settings, VALID_MOODS

logger = logging.getLogger(__name__)

# Interpretation prompt template
INTERPRETATION_PROMPT = """Interpret these song lyrics. Return a JSON object with NO other text.

The JSON must have exactly this structure:
{{
  "mood": "<one of: joyful, romantic, melancholic, intense, peaceful, nostalgic, defiant, playful>",
  "summary": "<2-3 poetic sentences capturing what this song is truly about - its emotional core and universal message>",
  "lyrics": [<array of lyric interpretations>]
}}

For each lyric in the "lyrics" array, provide these 4 fields:
- "original": exact original text
- "romanized": phonetic pronunciation in English letters (ONLY if original uses non-Latin script like Arabic, Chinese, Japanese, Korean, Hindi, Cyrillic, etc. If already Latin script, use EMPTY "")
- "translation": natural English translation (capture the feeling, not word-by-word)
- "meaning": cultural interpretation in 20-30 words (emotion, metaphors, cultural context)

CRITICAL: Output ONLY valid JSON. No markdown, no explanation, no preamble.

Example:
{{"mood":"romantic","summary":"A declaration of eternal love that transcends time and distance. The singer promises to remain devoted through all of life's storms.","lyrics":[{{"original":"Je t'aime","romanized":"","translation":"I love you","meaning":"A tender declaration..."}}]}}

Lyrics to interpret:
{segments}
"""

# Similar songs prompt template
SIMILAR_SONGS_PROMPT = """Based on this song, suggest 4 similar songs that the listener would enjoy.

Song: {song_title}
Mood: {mood}
{about}

Return ONLY a JSON array with NO other text. Each song should have:
- "title": Full song title with artist (e.g., "Shape of You - Ed Sheeran")  
- "reason": Brief 4-6 word reason why it's similar

Focus on:
- Similar mood/vibe
- Similar genre or style
- Similar language if it's not English
- Mix of popular and lesser-known gems

JSON array only:"""


def _try_extract_partial_json(text: str, expected_count: int) -> List[Dict[str, Any]]:
    """
    Try to extract individual JSON objects from a malformed response.
    
    Args:
        text: Raw response text.
        expected_count: Expected number of objects.
        
    Returns:
        List of successfully parsed interpretation dicts.
    """
    results = []
    
    # Try to find individual objects
    object_pattern = r'\{[^{}]*"romanized"[^{}]*"translation"[^{}]*"meaning"[^{}]*\}'
    matches = re.findall(object_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and 'translation' in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    # Pad with fallbacks if needed
    if results:
        while len(results) < expected_count:
            results.append({
                'romanized': '',
                'translation': '',
                'meaning': '(interpretation unavailable)'
            })
    
    return results


def _normalize_text(text: str) -> str:
    """
    Normalize text for matching by removing extra whitespace and lowercasing.
    """
    # Remove extra whitespace, normalize unicode, lowercase
    normalized = ' '.join(text.split()).lower().strip()
    return normalized


def _clean_json_response(response_text: str) -> str:
    """
    Clean JSON response by removing markdown and other artifacts.
    
    Args:
        response_text: Raw LLM response.
        
    Returns:
        Cleaned JSON string.
    """
    json_text = response_text
    
    # Remove markdown code blocks
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        parts = json_text.split("```")
        if len(parts) >= 2:
            json_text = parts[1]
    
    return json_text.strip()


def interpret_segments(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Interpret song lyrics with translations, romanization, and cultural meaning.
    
    Args:
        segments: List of segment dicts with 'text', 'start', 'end'.
        
    Returns:
        Tuple of (interpreted_segments, mood, summary).
    """
    client = Anthropic()
    
    # Filter to segments with actual text
    text_segments = [s for s in segments if s['text'].strip()]
    
    if not text_segments:
        for seg in segments:
            seg['romanized'] = ''
            seg['translation'] = '(no lyrics detected)'
            seg['meaning'] = ''
        return segments, "playful", ""
    
    # Deduplicate - only interpret unique lyrics
    unique_texts = []
    seen = set()
    for s in text_segments:
        text_lower = s['text'].strip().lower()
        if text_lower not in seen:
            unique_texts.append(s['text'])
            seen.add(text_lower)
    
    segments_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(unique_texts)])
    max_retries = settings.api.max_retries
    interpretations = []
    mood = "playful"
    summary = ""
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=settings.api.claude_model,
                max_tokens=settings.api.claude_max_tokens,
                messages=[{
                    "role": "user", 
                    "content": INTERPRETATION_PROMPT.format(segments=segments_text)
                }]
            )
            
            response_text = response.content[0].text
            json_text = _clean_json_response(response_text)
            parsed = json.loads(json_text)
            
            if isinstance(parsed, dict) and 'lyrics' in parsed:
                mood = parsed.get('mood', 'playful')
                summary = parsed.get('summary', '')
                interpretations = parsed.get('lyrics', [])
            elif isinstance(parsed, list):
                interpretations = parsed
            else:
                raise json.JSONDecodeError("Unexpected format", json_text, 0)
            
            break
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                # Try partial recovery
                interpretations = _try_extract_partial_json(response_text, len(unique_texts))
                if not interpretations:
                    interpretations = [
                        {
                            'romanized': '',
                            'translation': text if i > 0 else '⚠️ AI interpretation failed',
                            'meaning': '' if i > 0 else 'Try again or choose a different song.'
                        }
                        for i, text in enumerate(unique_texts)
                    ]
                break
                
        except Exception as e:
            logger.error(f"Interpretation error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    
    # Build lookup for interpretations - match by "original" field, not by index
    interp_lookup = {}
    for i, interp in enumerate(interpretations):
        if isinstance(interp, dict):
            # Use Claude's "original" field to match, with fallback to index
            original_text = interp.get('original', '')
            if original_text:
                key = _normalize_text(original_text)
            elif i < len(unique_texts):
                key = _normalize_text(unique_texts[i])
            else:
                continue
            
            interp_lookup[key] = {
                'romanized': interp.get('romanized', ''),
                'translation': interp.get('translation', original_text or ''),
                'meaning': interp.get('meaning', '')
            }
        elif i < len(unique_texts):
            # Fallback for non-dict responses
            interp_lookup[_normalize_text(unique_texts[i])] = {
                'romanized': '',
                'translation': unique_texts[i],
                'meaning': '(interpretation unavailable)'
            }
    
    # Apply to ALL segments (including duplicates)
    result = []
    for seg in segments:
        text_key = _normalize_text(seg['text'])
        
        # Try exact match first
        if text_key in interp_lookup:
            interp = interp_lookup[text_key]
        else:
            # Try fuzzy match - find key that starts with same words or is contained
            interp = None
            for lookup_key, lookup_val in interp_lookup.items():
                # Check if one contains the other (handles minor variations)
                if text_key in lookup_key or lookup_key in text_key:
                    interp = lookup_val
                    break
                # Check if first 3 words match
                text_words = text_key.split()[:3]
                lookup_words = lookup_key.split()[:3]
                if text_words and text_words == lookup_words:
                    interp = lookup_val
                    break
        
        if interp:
            seg['romanized'] = interp['romanized']
            seg['translation'] = interp['translation']
            seg['meaning'] = interp['meaning']
        else:
            logger.debug(f"No match found for: '{seg['text'][:50]}...'")
            seg['romanized'] = ''
            seg['translation'] = seg['text']
            seg['meaning'] = ''
        result.append(seg)
    
    # Validate mood
    if mood not in VALID_MOODS:
        mood = "playful"
    
    matched = sum(1 for r in result if r.get('meaning'))
    logger.info(f"Interpreted {len(unique_texts)} unique lyrics, matched {matched}/{len(result)} segments, mood: {mood}")
    return result, mood, summary


def get_similar_songs(
    song_title: str, 
    mood: str = "playful", 
    summary: str = ""
) -> List[Dict[str, str]]:
    """
    Get similar song suggestions based on the current song.
    
    Args:
        song_title: Title of the current song.
        mood: Detected mood of the song.
        summary: Song summary/description.
        
    Returns:
        List of dicts with 'title' and 'reason' keys.
    """
    if not song_title:
        return []
    
    client = Anthropic()
    
    about = f'About: {summary}' if summary else ''
    prompt = SIMILAR_SONGS_PROMPT.format(
        song_title=song_title,
        mood=mood,
        about=about
    )
    
    try:
        response = client.messages.create(
            model=settings.api.claude_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = _clean_json_response(response.content[0].text)
        suggestions = json.loads(response_text)
        
        if isinstance(suggestions, list) and len(suggestions) > 0:
            return suggestions[:settings.ui.max_similar_songs]
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse similar songs response: {e}")
    except Exception as e:
        logger.error(f"Error getting similar songs: {e}")
    
    return []
