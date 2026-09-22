from __future__ import annotations

import unittest
from pathlib import Path

from src.main import build_predictor

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.txt"


class CorpusIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor, _ = build_predictor(CORPUS)

    def test_assignment_example_predicts_estatistica(self) -> None:
        prediction = self.predictor.predict_word("O aluno estudou para a prova de")
        self.assertEqual(prediction.word, "estatística")
        self.assertGreater(prediction.confidence_pct, 20.0)

    def test_labeled_collocations(self) -> None:
        cases = {
            "A análise exploratória de": "dados",
            "O teorema de": "bayes",
            "A probabilidade condicional de uma": "palavra",
            "O modelo utiliza n": "grams",
            "A suavização de": "laplace",
            "O corpus passou por": "limpeza",
            "A próxima palavra mais": "provável",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(self.predictor.predict_word(phrase).word, expected)


if __name__ == "__main__":
    unittest.main()
