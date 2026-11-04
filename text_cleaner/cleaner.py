import re


class TextCleaner:
    """Nettoyage et standardisation du texte pour l'analyse NLP."""

    @staticmethod
    def clean_text_leger(text: str) -> str:
        """Nettoyage basique : passage en minuscules et réduction des espaces multiples."""
        if not text:
            return ""
        return " ".join(text.lower().split())

    @staticmethod
    def clean_text(text: str, preserve_compound_words: bool = False, remove_isolated_numbers: bool = True) -> str:
        """
        Nettoie et normalise le texte français :
        - Conversion en minuscules
        - Unification des apostrophes
        - Suppression des marqueurs Gutenberg (_mot_)
        - Conservation optionnelle des mots composés (tiret ou underscore)
        - Élimination optionnelle des chiffres isolés
        - Suppression de la ponctuation résiduelle
        """
        if not text:
            return ""

        # 1. Minuscules et apostrophes standard
        resultat = text.lower()
        resultat = re.sub(r"['’`]", "'", resultat)

        # 2. Retrait des italiques Gutenberg et espaces
        resultat = re.sub(r'_([^_]+)_', r'\1', resultat)
        resultat = re.sub(r'\s+', ' ', resultat)

        # 3. Gestion des mots composés
        pattern_lettres = r'[a-zàâçéèêëîïôûùüÿñæœ]'
        if preserve_compound_words:
            resultat = re.sub(f'(?<={pattern_lettres})-(?={pattern_lettres})', '_', resultat)

        # 4. Suppression des nombres isolés
        if remove_isolated_numbers:
            resultat = re.sub(r'\b\d+\b', ' ', resultat)

        # 5. Filtrage des caractères autorisés
        caracteres_valides = "a-zàâçéèêëîïôûùüÿñæœ'"
        if preserve_compound_words:
            caracteres_valides += "_"
        else:
            caracteres_valides += "-"

        if not remove_isolated_numbers:
            caracteres_valides += "0-9"

        resultat = re.sub(f"[^{caracteres_valides}]+", " ", resultat)

        return resultat.strip()
