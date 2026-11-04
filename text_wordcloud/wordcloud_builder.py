from wordcloud import WordCloud


class WordCloudBuilder:
    """Création et configuration de nuages de mots."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 600,
        background_color: str = "white",
        colormap: str = "Blues",
        max_words: int = 150
    ):
        self.config = {
            "width": width,
            "height": height,
            "background_color": background_color,
            "colormap": colormap,
            "collocations": False,
            "max_words": max_words
        }

    def from_tokens(self, tokens: list[str], **overrides) -> WordCloud:
        """Génère l'objet WordCloud à partir d'une liste de mots."""
        options = {**self.config, **overrides}
        texte = " ".join(tokens)
        return WordCloud(**options).generate(texte)
