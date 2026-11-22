# cache_manager.py
"""
Persistent disk-based response and extraction caching module.
Stores extraction and QA results on persistent /data volume (or local ./my_cache)
to eliminate duplicate model inferences and save ZeroGPU quota.
"""
import os
import hashlib

CACHE_DIR = "/data/my_cache" if os.path.exists("/data") else "./my_cache"

try:
    from diskcache import Cache
    HAS_DISKCACHE = True
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception:
        pass
    cache = Cache(CACHE_DIR)
except Exception:
    HAS_DISKCACHE = False
    _memory_cache = {}

def make_key(file_name: str, question: str) -> str:
    """Generates deterministic MD5 key from filename and query string."""
    text = (file_name or "file") + "||" + (question or "default")
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_from_cache(file_name: str, question: str):
    """Retrieves cached result if present, else returns None."""
    key = make_key(file_name, question)
    if HAS_DISKCACHE:
        try:
            return cache.get(key)
        except Exception:
            return None
    return _memory_cache.get(key)

def save_to_cache(file_name: str, question: str, answer):
    """Persists result to cache."""
    key = make_key(file_name, question)
    if HAS_DISKCACHE:
        try:
            cache.set(key, answer, expire=86400 * 7)  # 7 days TTL
        except Exception:
            pass
    else:
        _memory_cache[key] = answer
