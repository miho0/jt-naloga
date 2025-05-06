import json
import string
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')

SUPPORTED_EMOTIONS = ["anger", "fear", "sadness", "joy"]
STOPWORDS = set(stopwords.words("english"))
SAMPLES_PER_CLASS = 26000  # ker jih mamo 26k za fear jih vec nemoremo
# unigrami - 5-grami
# lahko se spilamo malo s totimi vrednostmi
MIN_N_GRAMS = 1
MAX_N_GRAMS = 5
NUM_FEATURES = 30000


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = nltk.word_tokenize(text)
    return " ".join([t for t in tokens if t not in STOPWORDS])


def load_balanced_dataset(file_path, samples_per_class):
    texts = []
    labels = []
    counts = defaultdict(int)

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            emotion = data.get("emotion")

            if emotion not in SUPPORTED_EMOTIONS:
                continue

            if counts[emotion] >= samples_per_class:
                continue

            cleaned = clean_text(data["text"])
            texts.append(cleaned)
            labels.append(emotion)
            counts[emotion] += 1

            if all(counts[e] >= samples_per_class for e in SUPPORTED_EMOTIONS):
                break

    return texts, labels


def main():
    X, y = load_balanced_dataset("archive/dataset.json", SAMPLES_PER_CLASS)

    print(f"Loaded {len(X)} samples ({SAMPLES_PER_CLASS} per class)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(ngram_range=(MIN_N_GRAMS, MAX_N_GRAMS), max_features=NUM_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
