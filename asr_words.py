#!/usr/bin/env python3
"""Self-contained Parakeet-TDT token -> word -> sentence helpers (stdlib only).

`burst_captions.py` needs three things from the ASR stage:

* **words with timestamps** — so a detected vocal burst can be written into the
  script *between the two words nearest the burst*, and so silent gaps can be
  turned into ``[pause X.Xs]`` markers;
* **sentences with timestamps** — so each sentence gets its own delivery cue and
  its own bursts;
* a plain **sentence splitter** for the fallback path where the ASR returned no
  timestamps at all.

Historically these came from an out-of-repo helper directory (``ASR_HELPERS_DIR``),
which made the repo un-runnable for anyone else. The implementations below are
bundled so the pipeline runs from a clean checkout; ``burst_captions.py`` still
prefers the external helpers when that directory is importable.

Input format is exactly what ``ParakeetProcessor.decode(..., durations=...)``
returns: a list of ``{"token": str, "start": float, "end": float}`` with the
times already converted to **seconds**.
"""
import re

# Sentence-final punctuation, Latin + CJK + Arabic/Urdu.
_SENT_END = "\\.\\!\\?。！？…۔؟"
_SENT_SPLIT_RE = re.compile(rf"(?<=[{_SENT_END}])[\"'”’）\)\]]*\s+")
_TRAIL_RE = re.compile(rf"[{_SENT_END}\"'”’\s]+$")

# Scripts that are written without spaces between words; each character is its own
# "word" for timestamp purposes.
_CJK_RANGES = (
    (0x3040, 0x30FF),    # kana
    (0x3400, 0x4DBF),    # CJK ext A
    (0x4E00, 0x9FFF),    # CJK unified
    (0xF900, 0xFAFF),    # CJK compatibility
    (0xAC00, 0xD7AF),    # hangul syllables
)


def _is_cjk(ch):
    o = ord(ch)
    return any(a <= o <= b for a, b in _CJK_RANGES)


def split_sentences(text):
    """Split a transcript into sentences. Never returns an empty list for non-empty text."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts or [text]


def tokens_to_words(toks, scale=1.0):
    """Group Parakeet sub-word tokens into words with ``{"w", "start", "end"}``.

    A token opens a new word when it carries a leading space (SentencePiece marks
    word starts that way) or when it is a CJK character (no spaces in those
    scripts). Punctuation tokens — which the processor collapses to a zero-length
    span at the previous token's end — are glued onto the preceding word.
    """
    words = []
    for t in toks or []:
        tok = t.get("token", "")
        if tok is None or tok == "":
            continue
        s = float(t.get("start", 0.0)) * scale
        e = float(t.get("end", s)) * scale
        first = tok.lstrip()[:1]
        new_word = (not words) or tok[:1].isspace() or (first and _is_cjk(first))
        if new_word:
            w = tok.strip()
            if not w:
                continue
            words.append({"w": w, "start": s, "end": max(e, s)})
        else:
            words[-1]["w"] += tok
            words[-1]["end"] = max(words[-1]["end"], e, words[-1]["start"])
    return [w for w in words if w["w"]]


def _norm(s):
    """Lowercased, whitespace/punctuation-free form used only for alignment.

    ``\W`` is Unicode-aware, so CJK / Cyrillic / etc. letters survive and only
    punctuation and spacing are stripped."""
    return re.sub(r"[\W_]+", "", (s or "").lower(), flags=re.UNICODE)


def sentences_from_words(text, words):
    """Attach ``start``/``end`` to each sentence of ``text`` by aligning it to ``words``.

    Alignment is done on a punctuation-free, lowercased concatenation of the word
    strings, so ASR punctuation differences between the decoded transcript and the
    token stream cannot desynchronise it. Sentences that cannot be located keep
    ``start``/``end`` of ``None``; callers already handle that.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    if not words:
        return [dict(text=s, start=None, end=None) for s in sents]

    # concatenated normalised word text + a map from char position -> word index
    buf, owner = [], []
    for i, w in enumerate(words):
        n = _norm(w["w"])
        buf.append(n)
        owner.extend([i] * len(n))
    hay = "".join(buf)

    out, cur = [], 0
    for s in sents:
        needle = _norm(s)
        if not needle:
            out.append(dict(text=s, start=None, end=None))
            continue
        pos = hay.find(needle, cur)
        if pos < 0:
            # partial match: use as much of the sentence head as we can find
            for cut in range(len(needle), 3, -1):
                pos = hay.find(needle[:cut], cur)
                if pos >= 0:
                    needle = needle[:cut]
                    break
        if pos < 0:
            out.append(dict(text=s, start=None, end=None))
            continue
        i0, i1 = owner[pos], owner[min(pos + len(needle) - 1, len(owner) - 1)]
        out.append(dict(text=s, start=float(words[i0]["start"]), end=float(words[i1]["end"])))
        cur = pos + len(needle)
    return out


# Aliases matching the historical out-of-repo helper names.
_tokens_to_words = tokens_to_words
_sentences_from_words = sentences_from_words
