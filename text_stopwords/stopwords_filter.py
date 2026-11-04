from collections import Counter


class StopWordsFilter:
    """Filtrage des mots vides (stopwords) et des tokens trop courts."""

    # Mots vides courants de base en français
    DEFAUT_STOPWORDS = {
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "à", "au", "aux",
        "ce", "ces", "cette", "cet", "il", "ils", "elle", "elles", "je", "tu", "nous",
        "vous", "on", "se", "sa", "son", "ses", "leur", "leurs", "qui", "que", "quoi",
        "dont", "ou", "où", "mais", "donc", "or", "ni", "car", "pour", "par", "dans",
        "sur", "sous", "avec", "sans", "est", "sont", "été", "être", "avoir", "a", "ai",
        "plus", "pas", "ne", "comme", "tout", "tous", "toute", "toutes", "si", "y"
    }

    def __init__(self, sw_nltk: set = None, sw_spacy: set = None, extra: set = None):
        """Fusionne les dictionnaires de mots vides fournis."""
        self.stopwords = set()
        if sw_nltk:
            self.stopwords.update(s.lower() for s in sw_nltk)
        if sw_spacy:
            self.stopwords.update(s.lower() for s in sw_spacy)
        if extra:
            self.stopwords.update(s.lower() for s in extra)

        if not self.stopwords:
            self.stopwords = set(self.DEFAUT_STOPWORDS)

    def filter(self, tokens: list[str], min_len: int = 3) -> list[str]:
        """Élimine les stopwords et les tokens de longueur inférieure à min_len."""
        return [
            t for t in tokens
            if t.lower() not in self.stopwords and len(t) >= min_len
        ]

    def reduction_rate(self, tokens: list[str], min_len: int = 3) -> float:
        """Pourcentage de réduction en nombre de tokens après filtrage."""
        if not tokens:
            return 0.0
        conserves = len(self.filter(tokens, min_len=min_len))
        return round((1.0 - (conserves / len(tokens))) * 100, 2)

    def top_after(self, tokens: list[str], n: int = 20, min_len: int = 3) -> list[tuple[str, int]]:
        """Affiche les n tokens les plus fréquents après filtrage."""
        tokens_filtres = self.filter(tokens, min_len=min_len)
        return Counter(tokens_filtres).most_common(n)
