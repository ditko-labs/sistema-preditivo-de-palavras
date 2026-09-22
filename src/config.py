"""Constantes compartilhadas do modelo."""

from __future__ import annotations

UNK_TOKEN = "<unk>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"

DEFAULT_NGRAM_ORDER = 3
DEFAULT_LAPLACE_ALPHA = 1.0

# Quando o contexto da maior ordem foi visto, mistura com a ordem
# imediatamente abaixo. Assim o unigram não empurra artigos ("a", "o")
# por cima de uma continuação que o trigram já conhece.
HIGHER_ORDER_WEIGHT = 0.80
LOWER_ORDER_WEIGHT = 0.20

MIN_WORD_FREQUENCY = 1
DEFAULT_TOP_K = 1
HOLD_OUT_RATIO = 0.12
RANDOM_SEED = 42
