"""Coleta, normalização e tokenização do corpus."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_KEEP_LETTERS = re.compile(r"[^a-záàâãéêíóôõúçñü\s]", flags=re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    raw_text: str
    sentences: list[list[str]]

    @property
    def token_count(self) -> int:
        return sum(len(sentence) for sentence in self.sentences)


def load_corpus(path: str | Path) -> str:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus não encontrado: {corpus_path}")
    return corpus_path.read_text(encoding="utf-8")


def preprocess_text(text: str) -> CorpusDocument:
    """Normaliza o texto e devolve sentenças já tokenizadas."""
    normalized = _normalize(text)
    sentences = [
        tokens
        for sentence in _SENTENCE_SPLIT.split(normalized)
        if (tokens := tokenize(sentence))
    ]
    return CorpusDocument(raw_text=text, sentences=sentences)


def tokenize(text: str) -> list[str]:
    cleaned = _KEEP_LETTERS.sub(" ", text.casefold().replace("-", " "))
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned.split(" ") if cleaned else []


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())
