from __future__ import annotations

import unittest

from src.config import END_TOKEN, START_TOKEN, UNK_TOKEN
from src.ngrams import NGramModel


class NGramModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = [
            ["o", "aluno", "estudou"],
            ["o", "aluno", "leu"],
        ]
        self.model = NGramModel(order=3, min_word_frequency=1)
        self.model.fit(self.sentences)

    def test_vocab_excludes_boundary_tokens(self) -> None:
        self.assertEqual(self.model.vocab, {"o", "aluno", "estudou", "leu"})
        self.assertNotIn(START_TOKEN, self.model.vocab)
        self.assertNotIn(END_TOKEN, self.model.vocab)

    def test_unigram_and_bigram_counts(self) -> None:
        self.assertEqual(self.model.unigram_count("aluno"), 2)
        self.assertEqual(self.model.ngram_count(("o", "aluno")), 2)
        self.assertEqual(self.model.ngram_count(("aluno", "estudou")), 1)

    def test_maps_unknown_words_to_unk(self) -> None:
        self.assertEqual(self.model.map_tokens(["o", "xyz"]), ["o", UNK_TOKEN])

    def test_rejects_invalid_order(self) -> None:
        with self.assertRaises(ValueError):
            NGramModel(order=0).fit(self.sentences)


if __name__ == "__main__":
    unittest.main()
