"""Predição da próxima palavra via Teorema de Bayes com Laplace smoothing."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    DEFAULT_LAPLACE_ALPHA,
    DEFAULT_NGRAM_ORDER,
    DEFAULT_TOP_K,
    HIGHER_ORDER_WEIGHT,
    LOWER_ORDER_WEIGHT,
    START_TOKEN,
)
from src.ngrams import NGramModel
from src.preprocessing import tokenize


@dataclass(frozen=True, slots=True)
class Prediction:
    word: str
    probability: float

    @property
    def confidence_pct(self) -> float:
        return self.probability * 100.0


class BayesianWordPredictor:
    """Estima P(W_n | contexto) com prior, verossimilhança e evidência suavizados.

    P(W_n | contexto) = P(contexto | W_n) * P(W_n) / P(contexto)

    Laplace (add-α) entra em cada termo para que contextos inéditos e
    palavras raras não gerem probabilidade zero. A evidência P(contexto)
    é obtida por marginalização: soma dos scores não normalizados no
    vocabulário.

    A ordem usada é a maior cujo contexto foi observado. Uma fração
    menor da ordem imediatamente abaixo entra só para não ficar rígido
    demais; o unigram não compete quando o trigram já conhece o par.
    """

    def __init__(
        self,
        ngram_order: int = DEFAULT_NGRAM_ORDER,
        alpha: float = DEFAULT_LAPLACE_ALPHA,
        higher_order_weight: float = HIGHER_ORDER_WEIGHT,
        lower_order_weight: float = LOWER_ORDER_WEIGHT,
    ) -> None:
        if alpha <= 0:
            raise ValueError("O parâmetro de Laplace (alpha) deve ser positivo.")
        if ngram_order < 1:
            raise ValueError("A ordem do n-gram deve ser pelo menos 1.")
        if abs(higher_order_weight + lower_order_weight - 1.0) > 1e-9:
            raise ValueError("Os pesos de backoff devem somar 1.0.")

        self.alpha = alpha
        self.ngram_order = ngram_order
        self.higher_order_weight = higher_order_weight
        self.lower_order_weight = lower_order_weight
        self.model = NGramModel(order=max(ngram_order, 3))

    def fit(self, sentences: list[list[str]]) -> None:
        self.model.fit(sentences)

    def predict(self, phrase: str, top_k: int = DEFAULT_TOP_K) -> list[Prediction]:
        tokens = self.model.map_tokens(tokenize(phrase))
        ranked = self._rank(tokens)
        return ranked[: max(1, top_k)]

    def predict_word(self, phrase: str) -> Prediction:
        return self.predict(phrase, top_k=1)[0]

    def next_word_distribution(self, tokens: list[str]) -> dict[str, float]:
        return {item.word: item.probability for item in self._rank(tokens)}

    def _rank(self, tokens: list[str]) -> list[Prediction]:
        mapped = self.model.map_tokens(tokens)
        scores = {word: 0.0 for word in self.model.candidates}

        for weight, context in self._active_layers(mapped):
            posterior = self._posterior_over_vocab(context)
            for word, probability in posterior.items():
                scores[word] += weight * probability

        return [
            Prediction(word=word, probability=probability)
            for word, probability in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

    def _active_layers(self, tokens: list[str]) -> list[tuple[float, tuple[str, ...]]]:
        """Escolhe a maior ordem com contexto observado e, se houver, a imediatamente abaixo."""
        observed: list[tuple[str, ...]] = []
        for context_size in (2, 1, 0):
            context = self._context(tokens, context_size)
            if context_size == 0 or self.model.ngram_count(context) > 0:
                observed.append(context)

        if len(observed) == 1:
            return [(1.0, observed[0])]
        return [
            (self.higher_order_weight, observed[0]),
            (self.lower_order_weight, observed[1]),
        ]

    def _posterior_over_vocab(self, context: tuple[str, ...]) -> dict[str, float]:
        """Aplica Bayes e normaliza pela evidência no suporte observado.

        Palavras que nunca seguiram o contexto ficam de fora do suporte.
        Laplace continua valendo entre as continuações vistas, sem diluir
        a confiança em centenas de tokens impossíveis neste histórico.
        """
        support = self._support(context)
        unnormalized = {
            word: self._likelihood(context, word) * self._prior(word)
            for word in support
        }
        evidence = sum(unnormalized.values())
        posterior = {word: 0.0 for word in self.model.candidates}
        if evidence <= 0:
            uniform = 1.0 / max(len(support), 1)
            for word in support:
                posterior[word] = uniform
            return posterior
        for word, score in unnormalized.items():
            posterior[word] = score / evidence
        return posterior

    def _support(self, context: tuple[str, ...]) -> list[str]:
        if not context:
            return self.model.candidates
        observed = [
            word
            for word in self.model.candidates
            if self.model.ngram_count((*context, word)) > 0
        ]
        return observed or self.model.candidates

    def _prior(self, word: str) -> float:
        """P(W_n) com Laplace: (C(w) + α) / (N + αV)."""
        return (self.model.unigram_count(word) + self.alpha) / (
            self.model.total_tokens + self.alpha * self.model.vocab_size
        )

    def _likelihood(self, context: tuple[str, ...], word: str) -> float:
        """P(contexto | W_n) com Laplace: (C(ctx, w) + α) / (C(w) + α).

        O denominador C(w)+α é o mesmo numerador do prior. O produto
        P(ctx|w) P(w) fica proporcional a C(ctx, w)+α; depois da
        evidência isso é o n-gram add-α, sem o unigram de "a"/"o"
        atropelar uma continuação que de fato ocorreu.
        """
        if not context:
            return 1.0

        joint = self.model.ngram_count((*context, word))
        word_count = self.model.unigram_count(word)
        return (joint + self.alpha) / (word_count + self.alpha)

    def _context(self, tokens: list[str], size: int) -> tuple[str, ...]:
        if size <= 0:
            return tuple()
        if len(tokens) >= size:
            return tuple(tokens[-size:])
        padding = (START_TOKEN,) * (size - len(tokens))
        return padding + tuple(tokens)
