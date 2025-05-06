import json
import string
from collections import defaultdict
from time import time

import nltk
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

SUPPORTED_EMOTIONS = ["anger", "fear", "sadness", "joy"]
STOPWORDS = set(stopwords.words("english"))
SAMPLES_PER_CLASS = 20000  # Povečal sem število vzorcev za boljše učenje
MIN_N_GRAMS = 1
MAX_N_GRAMS = 5  # Zmanjšal sem, ker so višji n-grami redko uporabni
NUM_FEATURES = 30000
MAX_ITER = 1000

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        return " ".join([self.lemmatizer.lemmatize(t) for t in tokens if t not in STOPWORDS])

def load_balanced_dataset(file_path, samples_per_class):
    texts = []
    labels = []
    counts = defaultdict(int)
    preprocessor = TextPreprocessor()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            emotion = data.get("emotion")

            if emotion not in SUPPORTED_EMOTIONS:
                continue

            if counts[emotion] >= samples_per_class:
                continue

            cleaned = preprocessor.clean_text(data["text"])
            texts.append(cleaned)
            labels.append(emotion)
            counts[emotion] += 1

            if all(counts[e] >= samples_per_class for e in SUPPORTED_EMOTIONS):
                break

    return texts, labels

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining {model_name}...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    
    start_time = time()
    y_pred = model.predict(X_test)
    predict_time = time() - start_time
    
    print(f"Training time: {train_time:.2f}s, Prediction time: {predict_time:.2f}s")
    print(classification_report(y_test, y_pred, digits=3))

def main():
    # Load and preprocess data
    X, y = load_balanced_dataset("archive/dataset.json", SAMPLES_PER_CLASS)
    print(f"Loaded {len(X)} samples ({SAMPLES_PER_CLASS} per class)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(MIN_N_GRAMS, MAX_N_GRAMS), max_features=NUM_FEATURES, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Define models to test
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=MAX_ITER,
            class_weight='balanced',
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5
        ),
        "Linear SVM": LinearSVC(
            max_iter=MAX_ITER,
            class_weight='balanced',
            dual=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=50,
            n_jobs=-1
        ),
        "Multilayer Perceptron": MLPClassifier(
            hidden_layer_sizes=(100,50),
            max_iter=MAX_ITER,
            early_stopping=True
        )
    }
    
    # Evaluate each model
    for name, model in models.items():
        evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test, name)
    
    # Try with SMOTE for handling class imbalance
    print("\nTrying SMOTE for handling class imbalance...")
    smote_pipeline = make_imb_pipeline(
        vectorizer,
        SMOTE(random_state=42),
        LogisticRegression(max_iter=MAX_ITER, solver='saga')
    )
    evaluate_model(smote_pipeline, X_train, X_test, y_train, y_test, "LogReg with SMOTE")

if __name__ == "__main__":
    main()