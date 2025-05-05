import collections
import json
import string

import nltk
from nltk import ngrams

# https://www.kaggle.com/datasets/devdope/900k-spotify

nltk.download("punkt_tab")

SUPPORTED_EMOTIONS = ["anger", "fear", "sadness", "joy"]
X_GRAMS = 4


def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()


def main():
    lyrics: dict[str, list[str]] = {}

    for emotion in SUPPORTED_EMOTIONS:
        lyrics[emotion] = []

    with open("dataset.json", "r", encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line)
            emotion = json_data["emotion"]

            if emotion not in SUPPORTED_EMOTIONS:
                continue

            lyrics[emotion].append(json_data["text"])

    for emotion, lyrics in lyrics.items():
        grams: collections.Counter = collections.Counter()

        for lyric in lyrics:
            tokens = nltk.word_tokenize(clean_text(lyric))
            songNgrams = ngrams(tokens, X_GRAMS)
            songNgramsStr = [" ".join(gram) for gram in list(songNgrams)]
            grams.update(songNgramsStr)

        sorted_dict = dict(sorted(grams.items(), key=lambda x: x[1], reverse=True))

        with open("new_no_punc_result_" + emotion + "_" + str(X_GRAMS) + ".json", "w") as fp:
            json.dump(sorted_dict, fp)


if __name__ == "__main__":
    main()
