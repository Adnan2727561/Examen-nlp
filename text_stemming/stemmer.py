from collections import defaultdict
from nltk.stem import SnowballStemmer


class Stemmer:
    """Racinisation (stemming) de mots français via l'algorithme Snowball."""

    def __init__(self, langue: str = "french"):
        self.outil = SnowballStemmer(langue)

    def stem(self, mot: str) -> str:
        """Retourne la racine d'un mot."""
        if not mot:
            return ""
        return self.outil.stem(mot.lower())

    def stem_all(self, tokens: list[str]) -> list[str]:
        """Applique la racinisation sur toute la liste de tokens."""
        return [self.stem(t) for t in tokens]

    def vocab_reduction(self, tokens: list[str]) -> float:
        """Calcule le pourcentage de compression du vocabulaire après racinisation."""
        if not tokens:
            return 0.0
        nb_avant = len(set(tokens))
        if nb_avant == 0:
            return 0.0
        nb_apres = len(set(self.stem_all(tokens)))
        return round((1.0 - (nb_apres / nb_avant)) * 100, 2)

    def families(self, tokens: list[str], top: int = 10) -> list[tuple[str, set[str]]]:
        """Regroupe les mots distincts partageant une même racine."""
        groupes = defaultdict(set)
        for mot in set(tokens):
            if mot:
                groupes[self.stem(mot)].add(mot)
        # Tri par nombre de variantes décroissant
        return sorted(groupes.items(), key=lambda elem: len(elem[1]), reverse=True)[:top]
