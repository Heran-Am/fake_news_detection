import re
import html
import unicodedata

# simple patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<.*?>")
WIRE_PREFIX_RE = re.compile(r"^\s*\(?reuters\)?\s*[-–—:]\s*", re.IGNORECASE)

def normalize_text(text: str, remove_reuters: bool = True) -> str:
    if text is None:
        return ""
    text = str(text)

    # Fix HTML entities (&amp; etc.)
    text = html.unescape(text)

    # Normalize unicode weirdness (quotes/dashes)
    text = unicodedata.normalize("NFKC", text)

    # Strip HTML tags if present
    text = HTML_TAG_RE.sub(" ", text)

    # Optional: remove Reuters wire prefix + token
    if remove_reuters:
        text = WIRE_PREFIX_RE.sub("", text)
        text = re.sub(r"\breuters\b", " ", text, flags=re.IGNORECASE)

    # Remove URLs (optional, but usually fine)
    text = URL_RE.sub(" ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text