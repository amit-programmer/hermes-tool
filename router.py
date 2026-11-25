# router.py
"""
Intelligent Query & Intent Router (CPU-Only).
Classifies requests within 0.005 seconds to route to the appropriate model or fast-path:
- Template fast-path: Greetings ('hi', 'hello', 'hey', 'namaste', etc.)
- Normal Chat: Short conversational queries -> Qwen2.5-1.5B
- Coding: Code generation, debugging, refactoring -> Qwen2.5-Coder-3B
- Logical: In-depth reasoning, math, why/how, analysis -> DeepSeek-R1-Distill-1.5B
- Heavy / Multi-step: Files, large context, heavy tasks -> Fallback GPU / DeepSeek

Scoring:
- Normal words / greetings: -3
- Hard / Analytical / Reasoning words: +5
- Coding keywords: +5 (triggers coding route)
- Has File attachment: +5 (triggers file route)
Score < 2 -> NORMAL, Score >= 2 -> HARD (CODING / LOGICAL)
"""
import re
from typing import Dict, Any, Optional

GREETING_WORDS = {
    "hi", "hello", "hey", "namaste", "hola", "greetings", 
    "good morning", "good afternoon", "good evening", "kaise ho",
    "kya haal hai", "who are you", "somi kaun ho", "tum kaun ho"
}

CODING_KEYWORDS = {
    "code", "python", "javascript", "typescript", "java", "c++", "rust",
    "function", "class", "def ", "def", "async", "await", "import",
    "bug", "error", "traceback", "exception", "api", "json", "endpoint",
    "html", "css", "sql", "database", "query", "regex", "algorithm",
    "script", "unittest", "pytest", "fastapi", "docker", "git", "syntax"
}

LOGICAL_KEYWORDS = {
    "kyu", "kaise", "why", "how", "compare", "logic", "reason", "analyze",
    "analysis", "explain", "samjhao", "strategy", "architecture", "solve",
    "proof", "calculate", "math", "derive", "evaluate", "optimize", "diff"
}

NORMAL_KEYWORDS = {
    "hi", "hello", "hey", "ok", "okay", "theek", "thanks", "dhanyawad",
    "yes", "ha", "nahi", "no", "bye", "alvida", "good"
}

def is_greeting(message: str) -> bool:
    """Detects simple conversational greeting in < 1ms."""
    clean = message.strip().lower()
    clean_words = re.findall(r"\b\w+\b", clean)
    if not clean_words:
        return False
    # If 1-5 words and contains any greeting phrase
    if len(clean_words) <= 5 and any(g in clean for g in GREETING_WORDS):
        return True
    return False

def identify_message(message: str, has_file: bool = False) -> Dict[str, Any]:
    """
    Evaluates message length, keyword density, and file presence.
    Returns:
        {
            "route": "greeting" | "normal" | "coding" | "logical" | "file_task",
            "model_type": "normal" | "coding" | "logical" | "heavy",
            "score": int,
            "is_greeting": bool,
            "has_file": bool,
            "reason": str
        }
    """
    clean = (message or "").strip().lower()
    words = re.findall(r"\b\w+\b", clean)
    word_count = len(words)

    # 1. Fast-Path: Greeting
    if is_greeting(message) and not has_file:
        return {
            "route": "greeting",
            "model_type": "none",
            "score": -5,
            "is_greeting": True,
            "has_file": False,
            "reason": "Direct greeting detected"
        }

    # 1.5 Heavy Input Check (> 500 words or > 100 lines of code)
    if word_count > 500 or len(message.splitlines()) > 100:
        return {
            "route": "heavy",
            "model_type": "coding",
            "score": 15,
            "is_greeting": False,
            "has_file": has_file,
            "reason": f"Heavy task detected ({word_count} words / {len(message.splitlines())} lines) -> GPU Qwen Coder"
        }

    score = 0
    detected_coding = []
    detected_logical = []
    detected_normal = []

    # 2. Length evaluation
    if word_count <= 5 and not has_file:
        score -= 2

    # 3. File presence
    if has_file:
        score += 5

    # 4. Keyword evaluation
    for w in words:
        if w in CODING_KEYWORDS:
            score += 5
            detected_coding.append(w)
        elif w in LOGICAL_KEYWORDS:
            score += 5
            detected_logical.append(w)
        elif w in NORMAL_KEYWORDS:
            score -= 3
            detected_normal.append(w)

    # Coding patterns check (e.g. code blocks, function calls)
    if "```" in message or "def " in message or "class " in message or "function(" in message:
        score += 5
        detected_coding.append("code_block")

    # Determine final route and model_type
    if has_file:
        route = "file_task"
        model_type = "logical"  # DeepSeek for reasoned chunk analysis
        reason = f"File attached (Score: {score})"
    elif len(detected_coding) > 0 and len(detected_coding) >= len(detected_logical):
        route = "coding"
        model_type = "coding"   # Qwen2.5-Coder-3B
        reason = f"Coding keywords: {list(set(detected_coding))[:3]} (Score: {score})"
    elif len(detected_logical) > 0 or score >= 2:
        route = "logical"
        model_type = "logical"  # DeepSeek-R1-Distill-1.5B
        reason = f"Logical/reasoning inquiry: {list(set(detected_logical))[:3]} (Score: {score})"
    else:
        route = "normal"
        model_type = "normal"   # Qwen2.5-1.5B
        reason = f"Conversational query (Score: {score})"

    return {
        "route": route,
        "model_type": model_type,
        "score": score,
        "is_greeting": False,
        "has_file": has_file,
        "reason": reason
    }
