"""
Script de démonstration séquentielle du pipeline NLP (Étapes 1 à 11).
Analyse du roman "Le Comte de Monte-Cristo".
"""

import os
import sys

# Support d'encodage pour terminaux Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from chargement_corpus import LoadRoman
from text_cleaner import TextCleaner
from text_tokenizer import Tokenizer
from text_frequency import FrequencyAnalyzer
from text_stopwords import StopWordsFilter
from text_stemming import Stemmer
from text_vectorizer import TextVectorizer


def run_pipeline():
    print("=" * 60)
    print("        PIPELINE NLP - LE COMTE DE MONTE-CRISTO")
    print("=" * 60)

    # 1. Chargement du corpus
    corpus_file = "monte_cristo.txt"
    if not os.path.exists(corpus_file):
        print(f"Téléchargement du roman depuis Gutenberg...")
        url = "https://www.gutenberg.org/files/17989/17989-0.txt"
        texte_brut = LoadRoman.load_roman(url)
        debut = "*** START OF THE PROJECT GUTENBERG EBOOK 17989 ***"
        fin = "*** END OF THE PROJECT GUTENBERG EBOOK 17989 ***"
        roman = LoadRoman.extraire_roman(texte_brut, debut, fin)
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(roman)
    else:
        print(f"Chargement du fichier local '{corpus_file}'...")
        roman = LoadRoman.load_local(corpus_file)

    print(f"-> Taille du texte brut : {len(roman):,} caractères.")
    print(f"-> Nombre de lignes : {roman.count(chr(10)):,}")

    # 2. Nettoyage du texte
    print("\n[Étape 3] Nettoyage et normalisation...")
    texte_propre = TextCleaner.clean_text(roman, preserve_compound_words=True, remove_isolated_numbers=True)
    print(f"-> Taille après nettoyage : {len(texte_propre):,} caractères.")

    # 3. Tokenisation
    print("\n[Étape 4] Tokenisation...")
    tokens = Tokenizer.tokenize_words(texte_propre)
    print(f"-> Nombre total de tokens : {len(tokens):,}")
    print(f"-> Richesse lexicale (TTR) : {Tokenizer.lexical_richness(tokens):.4f}")

    # 4. Analyse fréquentielle
    print("\n[Étape 5] Analyse des fréquences...")
    freq = FrequencyAnalyzer(tokens)
    print("-> Top 5 des mots les plus fréquents :")
    for mot, count in freq.top_n(5):
        print(f"   • {mot:<12} : {count} occurrences")
    hapax = freq.hapax_legomena()
    print(f"-> Nombre d'hapax : {len(hapax):,} ({freq.hapax_ratio():.2%} du vocabulaire)")

    # 5. Filtrage des mots vides
    print("\n[Étape 6] Filtrage des stopwords...")
    filtre = StopWordsFilter()
    tokens_filtres = filtre.filter(tokens, min_len=3)
    taux_red = filtre.reduction_rate(tokens, min_len=3)
    print(f"-> Tokens après filtrage : {len(tokens_filtres):,} (réduction de {taux_red:.1f}%)")
    print("-> Top 5 signifiant :")
    for mot, count in filtre.top_after(tokens, n=5, min_len=3):
        print(f"   • {mot:<12} : {count} occurrences")

    # 6. Racinisation (Stemming)
    print("\n[Étape 8] Racinisation (Snowball Stemmer)...")
    stemmer = Stemmer("french")
    compression = stemmer.vocab_reduction(tokens_filtres)
    print(f"-> Compression du vocabulaire après racinisation : {compression:.1f}%")

    # 7. Vectorisation et Recherche (BoW & TF-IDF)
    print("\n[Étapes 10-11] Vectorisation TF-IDF & Recherche documentaire...")
    # Découpage du roman en 4 sections pour tester la vectorisation documentaire
    taille_bloc = len(texte_propre) // 4
    blocs = [texte_propre[i:i + taille_bloc] for i in range(0, len(texte_propre), taille_bloc)][:4]
    noms = ["Section 1 (Début)", "Section 2 (Complot)", "Section 3 (Château d'If)", "Section 4 (Vengeance)"]

    vec = TextVectorizer(stop_words=list(filtre.stopwords), max_features=100)
    df_tfidf = vec.tfidf(blocs, noms)
    print(f"-> Matrice TF-IDF générée : {df_tfidf.shape[0]} documents × {df_tfidf.shape[1]} descripteurs.")

    requete = "dantes edmond chateau vengeance prison"
    print(f"-> Test de recherche pour la requête : '{requete}'")
    resultats = vec.search(requete, blocs, noms, top=2)
    for section, score in resultats:
        print(f"   ⭐ {section} (score similarité : {score})")

    print("\n" + "=" * 60)
    print("Traitement terminé avec succès.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
