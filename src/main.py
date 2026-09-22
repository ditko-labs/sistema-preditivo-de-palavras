"""Ponto de entrada: treina o modelo e prediz a próxima palavra."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.bayes import BayesianWordPredictor, Prediction
from src.config import DEFAULT_LAPLACE_ALPHA, DEFAULT_NGRAM_ORDER, DEFAULT_TOP_K
from src.evaluator import evaluate, format_report
from src.preprocessing import load_corpus, preprocess_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "corpus.txt"


def build_predictor(
    corpus_path: Path,
    ngram_order: int = DEFAULT_NGRAM_ORDER,
    alpha: float = DEFAULT_LAPLACE_ALPHA,
) -> tuple[BayesianWordPredictor, list[list[str]]]:
    document = preprocess_text(load_corpus(corpus_path))
    predictor = BayesianWordPredictor(ngram_order=ngram_order, alpha=alpha)
    predictor.fit(document.sentences)
    return predictor, document.sentences


def format_suggestion(prediction: Prediction) -> str:
    word = prediction.word[:1].upper() + prediction.word[1:]
    return f"Sugestão: {word}, Confiança: {prediction.confidence_pct:.2f}%"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    predictor, sentences = build_predictor(args.corpus, args.order, args.alpha)

    if args.evaluate:
        print(format_report(evaluate(predictor, sentences)))
        return 0

    if args.interactive:
        return _interactive_loop(predictor, args.top)

    if not args.frase:
        print("Informe uma frase ou use --interactive / --evaluate.", file=sys.stderr)
        return 2

    _print_predictions(predictor, args.frase, args.top)
    return 0


def _interactive_loop(predictor: BayesianWordPredictor, top_k: int) -> int:
    print("Sistema preditivo de palavras. Digite uma frase (ou 'sair').")
    while True:
        try:
            phrase = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if phrase.casefold() in {"sair", "exit", "quit"}:
            return 0
        if not phrase:
            continue
        _print_predictions(predictor, phrase, top_k)
    return 0


def _print_predictions(predictor: BayesianWordPredictor, phrase: str, top_k: int) -> None:
    for prediction in predictor.predict(phrase, top_k=top_k):
        print(format_suggestion(prediction))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sugere a próxima palavra de uma frase via Teorema de Bayes."
    )
    parser.add_argument("frase", nargs="?", help="Frase cujo próximo token será predito")
    parser.add_argument("-i", "--interactive", action="store_true", help="Modo interativo")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Roda a validação")
    parser.add_argument("-k", "--top", type=int, default=DEFAULT_TOP_K, help="Quantidade de sugestões")
    parser.add_argument("-n", "--order", type=int, default=DEFAULT_NGRAM_ORDER, help="Ordem máxima do n-gram")
    parser.add_argument("-a", "--alpha", type=float, default=DEFAULT_LAPLACE_ALPHA, help="Parâmetro de Laplace")
    parser.add_argument(
        "-c",
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Caminho do corpus textual",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
