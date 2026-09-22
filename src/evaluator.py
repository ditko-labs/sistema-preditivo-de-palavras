"""Validação do preditor: frases rotuladas e hold-out no próprio corpus."""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.bayes import BayesianWordPredictor
from src.config import HOLD_OUT_RATIO, RANDOM_SEED


@dataclass(frozen=True, slots=True)
class CaseResult:
    phrase: str
    expected: str
    predicted: str
    confidence_pct: float
    hit_top1: bool
    hit_top3: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    labeled: list[CaseResult]
    holdout: list[CaseResult]

    @property
    def labeled_accuracy(self) -> float:
        return _accuracy(self.labeled, top3=False)

    @property
    def labeled_top3(self) -> float:
        return _accuracy(self.labeled, top3=True)

    @property
    def holdout_accuracy(self) -> float:
        return _accuracy(self.holdout, top3=False)

    @property
    def holdout_top3(self) -> float:
        return _accuracy(self.holdout, top3=True)


LABELED_CASES: list[tuple[str, str]] = [
    ("O aluno estudou para a prova de", "estatística"),
    ("A análise exploratória de", "dados"),
    ("O teorema de", "bayes"),
    ("A probabilidade condicional de uma", "palavra"),
    ("O modelo utiliza n", "grams"),
    ("A suavização de", "laplace"),
    ("O corpus passou por", "limpeza"),
    ("A próxima palavra mais", "provável"),
    ("O professor explicou o teorema de", "bayes"),
    ("Os alunos fizeram a prova de", "estatística"),
]


def evaluate(
    predictor: BayesianWordPredictor,
    sentences: list[list[str]],
    labeled_cases: list[tuple[str, str]] | None = None,
    hold_out_ratio: float = HOLD_OUT_RATIO,
    seed: int = RANDOM_SEED,
) -> EvaluationReport:
    labeled = _run_labeled(predictor, labeled_cases or LABELED_CASES)
    train, test = _split_sentences(sentences, hold_out_ratio, seed)
    holdout_predictor = BayesianWordPredictor(
        ngram_order=predictor.ngram_order,
        alpha=predictor.alpha,
        higher_order_weight=predictor.higher_order_weight,
        lower_order_weight=predictor.lower_order_weight,
    )
    holdout_predictor.fit(train)
    holdout = _run_holdout(holdout_predictor, test)
    return EvaluationReport(labeled=labeled, holdout=holdout)


def format_report(report: EvaluationReport) -> str:
    lines = [
        "Validação do sistema preditivo",
        "",
        "Casos rotulados",
        f"  top-1: {_pct(report.labeled_accuracy)}  top-3: {_pct(report.labeled_top3)}  n={len(report.labeled)}",
    ]
    for case in report.labeled:
        mark = "ok" if case.hit_top1 else ("top3" if case.hit_top3 else "erro")
        lines.append(
            f"  [{mark}] '{case.phrase}' -> {case.predicted} "
            f"({case.confidence_pct:.2f}%)  esperado: {case.expected}"
        )

    lines.extend(
        [
            "",
            "Hold-out (última palavra de sentenças reservadas)",
            f"  top-1: {_pct(report.holdout_accuracy)}  top-3: {_pct(report.holdout_top3)}  n={len(report.holdout)}",
        ]
    )
    return "\n".join(lines)


def _run_labeled(
    predictor: BayesianWordPredictor,
    cases: list[tuple[str, str]],
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for phrase, expected in cases:
        ranked = predictor.predict(phrase, top_k=3)
        predicted = ranked[0]
        top3 = {item.word for item in ranked}
        results.append(
            CaseResult(
                phrase=phrase,
                expected=expected,
                predicted=predicted.word,
                confidence_pct=predicted.confidence_pct,
                hit_top1=predicted.word == expected,
                hit_top3=expected in top3,
            )
        )
    return results


def _split_sentences(
    sentences: list[list[str]],
    ratio: float,
    seed: int,
) -> tuple[list[list[str]], list[list[str]]]:
    eligible_idx = [i for i, sentence in enumerate(sentences) if len(sentence) >= 4]
    if not eligible_idx:
        return sentences, []

    rng = random.Random(seed)
    sample_size = max(1, int(len(eligible_idx) * ratio))
    test_idx = set(rng.sample(eligible_idx, sample_size))
    train = [sentence for i, sentence in enumerate(sentences) if i not in test_idx]
    test = [sentences[i] for i in sorted(test_idx)]
    return train, test


def _run_holdout(
    predictor: BayesianWordPredictor,
    sentences: list[list[str]],
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for sentence in sentences:
        expected = sentence[-1]
        phrase = " ".join(sentence[:-1])
        ranked = predictor.predict(phrase, top_k=3)
        predicted = ranked[0]
        top3 = {item.word for item in ranked}
        results.append(
            CaseResult(
                phrase=phrase,
                expected=expected,
                predicted=predicted.word,
                confidence_pct=predicted.confidence_pct,
                hit_top1=predicted.word == expected,
                hit_top3=expected in top3,
            )
        )
    return results


def _accuracy(results: list[CaseResult], top3: bool) -> float:
    if not results:
        return 0.0
    hits = sum(case.hit_top3 if top3 else case.hit_top1 for case in results)
    return hits / len(results)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"
