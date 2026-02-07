"""
Application settings and configuration.
Centralizes all magic strings, numbers, and configurable parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class APISettings:
    """API configuration."""
    whisper_model: str = "whisper-1"
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 8000
    max_retries: int = 3
    request_timeout: int = 30


@dataclass
class AudioSettings:
    """Audio processing configuration."""
    max_duration_seconds: int = 600  # 10 minutes
    audio_quality: str = "9"  # yt-dlp quality (9 = smallest)
    download_timeout: int = 120
    supported_formats: tuple = ("mp3", "webm", "m4a")
    # Optional Cobalt API base URL for YouTube audio (avoids yt-dlp JS/bot issues on Railway).
    # Set COBALT_API_URL in env if you run your own Cobalt instance (see https://github.com/imputnet/cobalt).
    cobalt_api_url: str = field(default_factory=lambda: os.environ.get("COBALT_API_URL", "").rstrip("/"))


@dataclass
class UISettings:
    """UI configuration."""
    player_height: int = 550
    max_search_results: int = 5
    max_similar_songs: int = 4
    thumbnail_width: int = 80


@dataclass
class CacheSettings:
    """Cache configuration."""
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
        ".cache"
    ))


# Mood color themes (primary accent, background gradient start, background gradient end)
MOOD_THEMES: Dict[str, Dict[str, str]] = {
    "joyful": {"accent": "#FFD700", "bg1": "#2d1f00", "bg2": "#1a1a00"},
    "romantic": {"accent": "#FF6B9D", "bg1": "#2d1a24", "bg2": "#1a0a14"},
    "melancholic": {"accent": "#6B9BFF", "bg1": "#0f1a2e", "bg2": "#0a1020"},
    "intense": {"accent": "#FF4444", "bg1": "#2e0f0f", "bg2": "#1a0808"},
    "peaceful": {"accent": "#6BFFB8", "bg1": "#0f2e1f", "bg2": "#081a10"},
    "nostalgic": {"accent": "#D4A574", "bg1": "#2e2010", "bg2": "#1a1408"},
    "defiant": {"accent": "#A855F7", "bg1": "#1f0f2e", "bg2": "#10081a"},
    "playful": {"accent": "#00D4FF", "bg1": "#0f1f2e", "bg2": "#081020"},
}

# Valid mood values
VALID_MOODS: List[str] = list(MOOD_THEMES.keys())

# Curated top songs by language for discovery
CURATED_SONGS: Dict[str, List[Dict[str, str]]] = {
    "🇫🇷 French": [
        {"title": "La Vie en Rose", "artist": "Édith Piaf", "query": "La Vie en Rose Edith Piaf official"},
        {"title": "Ne me quitte pas", "artist": "Jacques Brel", "query": "Ne me quitte pas Jacques Brel"},
        {"title": "Alors on danse", "artist": "Stromae", "query": "Stromae Alors on danse official"},
        {"title": "Je veux", "artist": "Zaz", "query": "Zaz Je veux official"},
    ],
    "🇪🇸 Spanish": [
        {"title": "Despacito", "artist": "Luis Fonsi", "query": "Despacito Luis Fonsi official video"},
        {"title": "Bésame Mucho", "artist": "Consuelo Velázquez", "query": "Besame Mucho original"},
        {"title": "La Bicicleta", "artist": "Shakira & Carlos Vives", "query": "Shakira Carlos Vives La Bicicleta official"},
        {"title": "Vivir Mi Vida", "artist": "Marc Anthony", "query": "Marc Anthony Vivir Mi Vida official"},
    ],
    "🇰🇷 Korean": [
        {"title": "Gangnam Style", "artist": "PSY", "query": "PSY Gangnam Style official"},
        {"title": "Dynamite", "artist": "BTS", "query": "BTS Dynamite official MV"},
        {"title": "How You Like That", "artist": "BLACKPINK", "query": "BLACKPINK How You Like That official"},
        {"title": "Love Scenario", "artist": "iKON", "query": "iKON Love Scenario official"},
    ],
    "🇯🇵 Japanese": [
        {"title": "Sukiyaki (Ue wo Muite Arukō)", "artist": "Kyu Sakamoto", "query": "Sukiyaki Kyu Sakamoto original"},
        {"title": "Lemon", "artist": "Kenshi Yonezu", "query": "Kenshi Yonezu Lemon official"},
        {"title": "Gurenge", "artist": "LiSA", "query": "LiSA Gurenge official"},
        {"title": "First Love", "artist": "Hikaru Utada", "query": "Hikaru Utada First Love"},
    ],
    "🇮🇳 Hindi/Urdu": [
        {"title": "Pasoori", "artist": "Ali Sethi & Shae Gill", "query": "Pasoori Coke Studio official"},
        {"title": "Tum Hi Ho", "artist": "Arijit Singh", "query": "Tum Hi Ho Arijit Singh official"},
        {"title": "Chaiyya Chaiyya", "artist": "Sukhwinder Singh", "query": "Chaiyya Chaiyya Dil Se official"},
        {"title": "Kal Ho Naa Ho", "artist": "Sonu Nigam", "query": "Kal Ho Naa Ho title track official"},
    ],
    "🇸🇦 Arabic": [
        {"title": "Ah W Noss", "artist": "Nancy Ajram", "query": "Nancy Ajram Ah W Noss official"},
        {"title": "3 Daqat", "artist": "Abu ft. Yousra", "query": "Abu 3 Daqat official"},
        {"title": "Tamally Maak", "artist": "Amr Diab", "query": "Amr Diab Tamally Maak official"},
        {"title": "Aatini Al Naya", "artist": "Fairuz", "query": "Fairuz Aatini Al Naya"},
    ],
    "🇧🇷 Portuguese": [
        {"title": "Garota de Ipanema", "artist": "Tom Jobim", "query": "Girl from Ipanema Tom Jobim original"},
        {"title": "Ai Se Eu Te Pego", "artist": "Michel Teló", "query": "Michel Telo Ai Se Eu Te Pego official"},
        {"title": "Magalenha", "artist": "Sergio Mendes", "query": "Sergio Mendes Magalenha"},
        {"title": "Mas Que Nada", "artist": "Jorge Ben Jor", "query": "Mas Que Nada Jorge Ben official"},
    ],
    "🇨🇳 Chinese": [
        {"title": "月亮代表我的心", "artist": "Teresa Teng", "query": "Teresa Teng Moon Represents My Heart"},
        {"title": "甜蜜蜜", "artist": "Teresa Teng", "query": "Teresa Teng Tian Mi Mi"},
        {"title": "告白气球", "artist": "Jay Chou", "query": "Jay Chou 告白气球 official"},
        {"title": "小幸运", "artist": "Hebe Tien", "query": "Hebe Tien 小幸运 official"},
    ],
    "🇹🇷 Turkish": [
        {"title": "Şımarık", "artist": "Tarkan", "query": "Tarkan Simarik official"},
        {"title": "Dudu", "artist": "Tarkan", "query": "Tarkan Dudu official"},
        {"title": "İstanbul İstanbul Olalı", "artist": "Sezen Aksu", "query": "Sezen Aksu Istanbul"},
    ],
    "🇮🇹 Italian": [
        {"title": "Con Te Partirò", "artist": "Andrea Bocelli", "query": "Andrea Bocelli Con Te Partiro official"},
        {"title": "Volare", "artist": "Domenico Modugno", "query": "Domenico Modugno Volare original"},
        {"title": "L'italiano", "artist": "Toto Cutugno", "query": "Toto Cutugno L'italiano"},
    ],
}


@dataclass
class Settings:
    """Main settings container."""
    api: APISettings = field(default_factory=APISettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    ui: UISettings = field(default_factory=UISettings)
    cache: CacheSettings = field(default_factory=CacheSettings)


# Global settings instance
settings = Settings()
