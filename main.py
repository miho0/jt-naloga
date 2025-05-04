import json
import collections
import string

def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

myEmotions = [
    "joy",
    "sadness",
    "anger",
    "fear"
]

class Song:
    def __init__(self, lyrics, emotion):
        self.lyrics = lyrics
        self.emotion = emotion

songs: list[Song] = []
groups = collections.defaultdict(list)

EMOTION_SIZE_LIMIT = 25000
GRAM_SIZE = 5

counted_songs = 0

with open("dataset.json", 'r', encoding='utf-8') as file:
    for line in file:
        try:
            json_data = json.loads(line)
            song = Song(clean_text(json_data["text"]), json_data["emotion"])

            if song.emotion not in myEmotions:
                continue

            if len(groups[song.emotion]) >= EMOTION_SIZE_LIMIT:
                continue

            groups[song.emotion].append(song)
            counted_songs += 1

            if counted_songs > EMOTION_SIZE_LIMIT * len(myEmotions):
                break

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

grams: dict[str, dict[str, int]] = {}

for emotion, songs in groups.items():
    tmp: dict[str, int] = {}

    for song in songs:
        words = song.lyrics.split()

        for i in range(len(words) - (GRAM_SIZE - 1)):
            gram = " ".join(words[i:i + GRAM_SIZE])

            if gram not in tmp:
                tmp[gram] = 0

            tmp[gram] += 1

    grams[emotion] = tmp

    sorted_dict = dict(sorted(tmp.items(), key=lambda x: x[1], reverse=True))

    with open("result_" + emotion + "_5.json", 'w') as fp:
        json.dump(sorted_dict, fp)
