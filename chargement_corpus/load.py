import os
import requests


class LoadRoman:
    """Gestion du chargement et de l'extraction du texte du roman."""

    @staticmethod
    def load_roman(url: str) -> str:
        """Télécharge le contenu brut depuis une URL avec encodage UTF-8."""
        try:
            req = requests.get(url, timeout=15)
            req.raise_for_status()
            req.encoding = "utf-8"
            return req.text
        except requests.RequestException as err:
            raise RuntimeError(f"Échec du téléchargement réseau : {err}")

    @staticmethod
    def load_local(filepath: str = "monte_cristo.txt") -> str:
        """Lit le texte depuis un fichier local."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier introuvable : {filepath}")
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def extraire_roman(texte: str, balise_debut: str = None, balise_fin: str = None) -> str:
        """Extrait le corps du roman en retirant l'en-tête et le pied de page."""
        contenu = texte.strip()

        if balise_debut:
            if balise_debut not in contenu:
                raise ValueError("Balise de début non trouvée dans le texte.")
            contenu = contenu.split(balise_debut, 1)[1]

        if balise_fin:
            if balise_fin not in contenu:
                raise ValueError("Balise de fin non trouvée dans le texte.")
            contenu = contenu.split(balise_fin, 1)[0]

        return contenu.strip()
