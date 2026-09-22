"""Contagem de n-grams e vocabulário a partir das sentenças do corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.config import END_TOKEN, MIN_WORD_FREQUENCY, START_TOKEN, UNK_TOKEN


@dataclass(slots=True)
class NGramModel:
    order: int
    min_word_frequency: int = MIN_WORD_FREQUENCY
    counts: dict[int, Counter[tuple[str, ...]]] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)
    total_tokens: int = 0

    def fit(self, sentences: list[list[str]]) -> None:
        if self.order < 1:
            raise ValueError("A ordem do n-gram deve ser pelo menos 1.")

        replaced = self._replace_rare_words(sentences)
        self.vocab = {token for sentence in replaced for token in sentence}
        self.counts = {n: Counter() for n in range(1, self.order + 1)}

        for sentence in replaced:
            padded = [START_TOKEN] * (self.order - 1) + sentence + [END_TOKEN]
            self.total_tokens += len(sentence)
            for n in range(1, self.order + 1):
                for gram in _sliding_window(padded, n):
                    self.counts[n][gram] += 1

    def ngram_count(self, gram: tuple[str, ...]) -> int:
        return self.counts.get(len(gram), Counter())[gram]

    def unigram_count(self, word: str) -> int:
        return self.counts[1][(word,)]

    def map_tokens(self, tokens: list[str]) -> list[str]:
        """Substitui palavras fora do vocabulário por UNK."""
        return [token if token in self.vocab else UNK_TOKEN for token in tokens]

    @property
    def candidates(self) -> list[str]:
        return sorted(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _replace_rare_words(self, sentences: list[list[str]]) -> list[list[str]]:
        frequencies = Counter(token for sentence in sentences for token in sentence)
        rare = {word for word, freq in frequencies.items() if freq < self.min_word_frequency}
        if not rare:
            return sentences

        mapped: list[list[str]] = []
        for sentence in sentences:
            mapped.append([UNK_TOKEN if token in rare else token for token in sentence])
        return mapped


def _sliding_window(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
