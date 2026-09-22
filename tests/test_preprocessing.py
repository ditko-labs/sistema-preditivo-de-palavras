from __future__ import annotations

import unittest

from src.preprocessing import preprocess_text, tokenize


class TokenizeTests(unittest.TestCase):
    def test_lowercase_and_strip_punctuation(self) -> None:
        self.assertEqual(tokenize("Olá, Mundo!"), ["olá", "mundo"])

    def test_keeps_portuguese_accents(self) -> None:
        self.assertEqual(tokenize("Estatística"), ["estatística"])

    def test_splits_hyphenated_terms(self) -> None:
        self.assertEqual(tokenize("n-grams"), ["n", "grams"])

    def test_empty_and_symbols_only(self) -> None:
        self.assertEqual(tokenize("   "), [])
        self.assertEqual(tokenize("..."), [])


class PreprocessTextTests(unittest.TestCase):
    def test_splits_sentences_and_drops_empty(self) -> None:
        document = preprocess_text("Primeira frase. Segunda frase! ??? Terceira?")
        self.assertEqual(
            document.sentences,
            [["primeira", "frase"], ["segunda", "frase"], ["terceira"]],
        )
        self.assertEqual(document.token_count, 5)


if __name__ == "__main__":
    unittest.main()
