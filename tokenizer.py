import re
import pickle
import os
from config import get_config

def _token_path(filename):
    cfg = get_config()
    return os.path.join(cfg["token_dir"], filename)

class Tokenizer:
    def __init__(self, file="tokens.pkl", mode="tokenize"):
        self.file = _token_path(file)
        if mode == "tokenize":
            with open(self.file, "rb") as f:
                self.tokens = pickle.load(f)
            with open(_token_path("inv_" + file), "rb") as s:
                self.inv_tokens = pickle.load(s)

    @staticmethod
    def split_text(text):
        # This regex splits words and keeps punctuation as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    @staticmethod
    def index_tokens(text, output_file="tokens.pkl"):
        out_path = _token_path(output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        words = Tokenizer.split_text(text)
        map = {}
        map_inv = []
        counter = 0
        for word in words:
            word = word.lower()
            if word not in map:
                map[word] = counter
                map_inv.append(word)
                counter += 1
        with open(out_path, "wb") as file:
            pickle.dump(map, file)
        with open(_token_path("inv_" + output_file), "wb") as file:
            pickle.dump(map_inv, file)
        return counter

    def tokenize_text(self, text):
        tokenized = []
        for word in text:
            try:
                tokenized.append(self.tokens[word.lower()])
            except KeyError:
                tokenized.append(-1)
        return tokenized

    def untokenize_text(self, text):
        tokenized = []
        length = len(self.tokens)
        for word in text:
            try:
                if(word == -1 or word > length-1):
                    tokenized.append('???')
                    continue
                tokenized.append(self.inv_tokens[word])
            except Exception:
                print("Error untokenizing: " + str(word))
        return tokenized


if __name__ == "__main__":
    cfg = get_config()
    corpus = cfg["corpus_file"]
    print(f"Indexing tokens from {corpus} -> {cfg['token_dir']}/")
    with open(corpus, "r", encoding="utf-8") as f:
        full_text = f.read()
    count = Tokenizer.index_tokens(full_text)
    print(f"Vocab size: {count}")

