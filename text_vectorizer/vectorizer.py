import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextVectorizer:
    """Représentations vectorielles BoW et TF-IDF avec recherche par similarité."""

    def __init__(
        self,
        stop_words: list[str] = None,
        min_df: int | float = 2,
        max_df: float = 0.90,
        max_features: int = 300
    ):
        self.stop_words = list(stop_words) if stop_words else None
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.bow_vec = None
        self.tfidf_vec = None

    def _build_params(self, n_docs: int) -> dict:
        """Adapte min_df et max_df pour éviter les erreurs sur de très petits échantillons."""
        ajust_min_df = 1 if n_docs < 3 else self.min_df
        ajust_max_df = 1.0 if n_docs < 3 else self.max_df
        return {
            "lowercase": True,
            "stop_words": self.stop_words,
            "min_df": ajust_min_df,
            "max_df": ajust_max_df,
            "max_features": self.max_features
        }

    def bow(self, blocs: list[str], noms: list[str]) -> pd.DataFrame:
        """Génère la matrice Bag of Words (comptages de termes) en DataFrame."""
        params = self._build_params(len(blocs))
        self.bow_vec = CountVectorizer(**params)
        matrice = self.bow_vec.fit_transform(blocs)
        return pd.DataFrame(
            matrice.toarray(),
            index=noms,
            columns=self.bow_vec.get_feature_names_out()
        )

    def tfidf(self, blocs: list[str], noms: list[str]) -> pd.DataFrame:
        """Génère la matrice pondérée TF-IDF en DataFrame."""
        params = self._build_params(len(blocs))
        self.tfidf_vec = TfidfVectorizer(
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            **params
        )
        matrice = self.tfidf_vec.fit_transform(blocs)
        return pd.DataFrame(
            matrice.toarray().round(4),
            index=noms,
            columns=self.tfidf_vec.get_feature_names_out()
        )

    def search(self, requete: str, blocs: list[str], noms: list[str], top: int = 3) -> list[tuple[str, float]]:
        """Recherche les documents les plus proches de la requête par similarité cosinus."""
        if not requete.strip() or not blocs:
            return []

        if self.tfidf_vec is None:
            self.tfidf(blocs, noms)

        mat_corpus = self.tfidf_vec.transform(blocs)
        mat_query = self.tfidf_vec.transform([requete])

        scores = cosine_similarity(mat_query, mat_corpus)[0]
        resultats = [(noms[i], round(float(score), 4)) for i, score in enumerate(scores)]
        return sorted(resultats, key=lambda x: x[1], reverse=True)[:top]
