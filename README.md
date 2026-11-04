# Pipeline NLP : Le Comte de Monte-Cristo

Traitement Automatique du Langage Naturel (NLP / TALN).  
Chaîne de traitement textuel, analyse statistique et représentations vectorielles (BoW & TF-IDF) sur le roman *Le Comte de Monte-Cristo*.

---

## 📁 Architecture du Projet

Chaque étape du traitement est organisée dans un module Python autonome :

```text
├── chargement_corpus/      # Étape 2  : Téléchargement et extraction du corps du texte
├── text_cleaner/           # Étape 3  : Nettoyage et normalisation (minuscules, regex)
├── text_tokenizer/         # Étape 4  : Segmentation en tokens, phrases et n-grammes
├── text_frequency/         # Étape 5  : Fréquences d'apparition, rangs, loi de Zipf, hapax
├── text_stopwords/         # Étape 6  : Suppression des mots vides
├── text_wordcloud/         # Étape 7  : Construction de nuages de mots
├── text_stemming/          # Étape 8  : Racinisation morphologique (Snowball)
├── text_lemmatizer/        # Étape 9  : Lemmatisation et catégories grammaticales (SpaCy)
├── text_vectorizer/        # Étapes 10-11 : Vectorisation BoW / TF-IDF et similarité cosinus
├── monte_cristo.txt        # Fichier texte du roman
├── rapport-nlp.ipynb       # Notebook Jupyter complet avec graphiques et visualisations
├── main.py                 # Script séquentiel exécutant les étapes du pipeline
└── requirements.txt        # Liste des dépendances Python requises
```

---

## ⚙️ Les Étapes du Pipeline

1. **Environnement** : Configuration des librairies (`nltk`, `spacy`, `scikit-learn`).
2. **Chargement** : Récupération du roman depuis Gutenberg ou fichier local.
3. **Nettoyage** : Assainissement du texte et traitement des mots composés et nombres isolés.
4. **Tokenisation** : Découpage en unités lexicales et calcul de richesse lexicale.
5. **Analyse Fréquentielle** : Distribution des fréquences, loi de Zipf et détection des hapax.
6. **Mots vides (Stopwords)** : Filtrage des mots non signifiants.
7. **Nuage de mots** : Représentation visuelle des termes clés.
8. **Racinisation (Stemming)** : Réduction morphologique via Snowball.
9. **Lemmatisation** : Analyse grammaticale (POS) et lemmes via SpaCy.
10. **Bag of Words (BoW)** : Matrice Document-Terme brute.
11. **TF-IDF & Similarité** : Matrice pondérée et recherche par similarité cosinus.

---

## 🚀 Installation

```bash
# 1. Création de l'environnement virtuel
python -m venv venv

# Activation sous Linux / macOS :
source venv/bin/activate
# Activation sous Windows (PowerShell) :
.\venv\Scripts\Activate.ps1

# 2. Installation des dépendances
pip install -r requirements.txt

# 3. Téléchargement des ressources linguistiques
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'stopwords')]"
python -m spacy download fr_core_news_sm
```

---

## ▶️ Exécution

### 1. En ligne de commande
Pour exécuter la démonstration séquentielle complète :
```bash
python main.py
```

### 2. Via le Notebook
Pour explorer l'analyse complète avec toutes les visualisations graphiques :
```bash
jupyter notebook rapport-nlp.ipynb
```
