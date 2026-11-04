from collections import Counter


class FrequencyAnalyzer:
    """Analyse statistique des fréquences des mots d'un corpus."""

    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self.total = len(self.tokens)
        self.freq = Counter(self.tokens)
        self._classement = None

    def top_n(self, n: int = 20) -> list[tuple[str, int]]:
        """Retourne les n tokens les plus fréquents avec leurs occurrences."""
        return self.freq.most_common(n)

    def top_n_pct(self, n: int = 20) -> list[tuple[str, int, float]]:
        """Retourne les n tokens avec occurrences et part relative en pourcentage."""
        if self.total == 0:
            return []
        return [(mot, nb, round((nb / self.total) * 100, 2)) for mot, nb in self.freq.most_common(n)]

    def frequency(self, mot: str) -> int:
        """Fréquence brute d'un mot donné (0 si absent)."""
        return self.freq.get(mot, 0)

    def rank(self, mot: str) -> int | None:
        """Rang d'un mot dans le classement décroissant des fréquences."""
        if self._classement is None:
            self._classement = [m for m, _ in self.freq.most_common()]
        try:
            return self._classement.index(mot) + 1
        except ValueError:
            return None

    def hapax_legomena(self) -> list[str]:
        """Liste des mots apparaissant une seule et unique fois."""
        return [mot for mot, count in self.freq.items() if count == 1]

    def hapax_ratio(self) -> float:
        """Proportion d'hapax par rapport au vocabulaire distinct (0 à 1)."""
        if not self.freq:
            return 0.0
        return len(self.hapax_legomena()) / len(self.freq)

    def zipf_data(self) -> tuple[list[int], list[int]]:
        """Données pour tracer la courbe de Zipf (rangs et fréquences associées)."""
        frequences_triees = sorted(self.freq.values(), reverse=True)
        rangs = list(range(1, len(frequences_triees) + 1))
        return rangs, frequences_triees
