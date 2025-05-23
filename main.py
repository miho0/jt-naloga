import json
import string

# https://www.kaggle.com/datasets/devdope/900k-spotify

SUPPORTED_EMOTIONS = ["anger", "fear", "sadness", "joy"]

# EMOTION_SIZE_LIMIT = 25000
GRAM_SIZE = 4


def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()


def main():
    lyrics: dict[str, list[str]] = {}

    for emotion in SUPPORTED_EMOTIONS:
        lyrics[emotion] = []

    counted_songs = 0

    with open("dataset.json", 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            emotion = json_data["emotion"]

            if emotion not in SUPPORTED_EMOTIONS:
                continue

            # if len(lyrics[emotion]) >= EMOTION_SIZE_LIMIT:
            #     continue

            lyrics[emotion].append(json_data["text"])
            counted_songs += 1

            # if counted_songs > EMOTION_SIZE_LIMIT * len(SUPPORTED_EMOTIONS):
            #     break

    for emotion, lyrics in lyrics.items():
        grams: dict[str, int] = {}

        for lyric in lyrics:
            words = lyric.split()

            for i in range(len(words) - (GRAM_SIZE - 1)):
                gram = " ".join(words[i:i + GRAM_SIZE])

                if gram not in grams:
                    grams[gram] = 0

                grams[gram] += 1

        sorted_dict = dict(sorted(grams.items(), key=lambda x: x[1], reverse=True))

        with open("result_" + emotion + "_5.json", 'w') as fp:
            json.dump(sorted_dict, fp)


if __name__ == "__main__":
    main()
