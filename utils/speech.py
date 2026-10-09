"""
Text-to-Speech (TTS) module using pyttsx3.
Provides safe, non-blocking audio narration for accumulated ASL text.
"""
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _speak_worker(text: str, rate: int = 150, volume: float = 1.0) -> None:
    """Internal worker thread that runs pyttsx3 safely."""
    try:
        import pyttsx3
        # Initialize a fresh engine instance in the worker thread for COM safety
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as exc:
        logger.warning(f"Text-to-speech error: {exc}")


def speak_text(text: str, async_mode: bool = True, rate: int = 150) -> Optional[threading.Thread]:
    """
    Narrates the provided text using pyttsx3.
    
    Args:
        text: The sentence or word to speak.
        async_mode: If True, runs on a separate daemon thread to avoid blocking the UI.
        rate: Speaking rate in words per minute (default: 150).
        
    Returns:
        The started Thread if async_mode is True, else None.
    """
    clean_text = str(text).strip()
    if not clean_text:
        return None

    if async_mode:
        thread = threading.Thread(
            target=_speak_worker,
            args=(clean_text, rate),
            daemon=True
        )
        thread.start()
        return thread
    else:
        _speak_worker(clean_text, rate)
        return None
