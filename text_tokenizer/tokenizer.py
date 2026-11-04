import re
import pandas as pd
from nltk import bigrams as nltk_bigrams, trigrams as nltk_trigrams
from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenizer:
    """Découpage du texte en mots, phrases et n-grammes."""

    @staticmethod
    def tokenize_words(text: str, language: str = "french") -> list[str]:
        """Découpage en tokens avec NLTK (ou fallback regex)."""
        if not text:
            return []
        try:
            return word_tokenize(text, language=language)
        except LookupError:
            return re.findall(r"[a-zA-Zàâçéèêëîïôûùüÿñæœ_'-]+", text)

    @staticmethod
    def tokenize_sentences(text: str, language: str = "french") -> list[str]:
        """Découpage du texte en phrases."""
        if not text:
            return []
        try:
            return sent_tokenize(text, language=language)
        except LookupError:
            return [p.strip() for p in re.split(r'[.!?]+', text) if p.strip()]

    @staticmethod
    def tokenize_spacy(text: str, nlp) -> list[str]:
        """Tokenisation linguistique fine avec SpaCy."""
        if not text:
            return []
        doc = nlp(text)
        return [tok.text for tok in doc if not tok.is_space]

    @staticmethod
    def get_bigrams(tokens: list[str]) -> list[tuple[str, str]]:
        """Génère les paires de mots consécutifs (bigrammes)."""
        return list(nltk_bigrams(tokens))

    @staticmethod
    def get_trigrams(tokens: list[str]) -> list[tuple[str, str, str]]:
        """Génère les triplets de mots consécutifs (trigrammes)."""
        return list(nltk_trigrams(tokens))

    @staticmethod
    def lexical_richness(tokens: list[str]) -> float:
        """Calcule le ratio de richesse lexicale (mots uniques / mots totaux)."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    @staticmethod
    def to_dataframe(tokens: list[str]) -> pd.DataFrame:
        """Construit un DataFrame avec position, token et longueur de mot."""
        return pd.DataFrame({
            "position": range(len(tokens)),
            "token": tokens,
            "longueur": [len(mot) for mot in tokens]
        })
