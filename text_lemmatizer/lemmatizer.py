from collections import Counter


class Lemmatizer:
    """Lemmatisation et étiquetage morpho-syntaxique (POS) via SpaCy."""

    def __init__(self, nlp):
        self.nlp = nlp

    def clean_lemmas(self, text: str, min_len: int = 3) -> list[str]:
        """Retourne les lemmes débarrassés des mots vides et de la ponctuation."""
        if not text:
            return []
        doc = self.nlp(text)
        return [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space and len(t.lemma_) >= min_len
        ]

    def by_pos(self, text: str, pos: str, min_len: int = 3) -> list[str]:
        """Filtre les lemmes selon une étiquette morpho-syntaxique (ex: 'NOUN', 'VERB', 'ADJ')."""
        if not text:
            return []
        doc = self.nlp(text)
        return [
            t.lemma_.lower()
            for t in doc
            if t.pos_ == pos and len(t.lemma_) >= min_len
        ]

    def pos_distribution(self, text: str) -> list[tuple[str, int]]:
        """Calcule la fréquence de chaque catégorie grammaticale présente dans le texte."""
        if not text:
            return []
        doc = self.nlp(text)
        return Counter(t.pos_ for t in doc if not t.is_space).most_common()
