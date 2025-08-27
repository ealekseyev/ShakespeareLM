import re
import pickle

class Tokenizer:
    def __init__(self, file="tokens.pkl", mode="tokenize"):
        self.file = file
        if mode == "tokenize":
            with open(self.file, "rb") as f:
                self.tokens = pickle.load(f)
            with open("inv_" + self.file, "rb") as s:
                self.inv_tokens = pickle.load(s)

    @staticmethod
    def split_text(text):
        # This regex splits words and keeps punctuation as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    @staticmethod
    def index_tokens(text, output_file="tokens.pkl"):
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
        with open(output_file, "wb") as file:
            pickle.dump(map, file)
        with open("inv_" + output_file, "wb") as file:
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
    t = Tokenizer(mode="tokenize")
    # st = Tokenizer.split_text("Once upon a time, there was a snfefj.")
    # tokenized = t.tokenize_text(st)
    # print(tokenized)
    # print(t.untokenize_text(tokenized))
    breakpoint()
    with open("shakespeare_cleaned.txt", "r") as ss:
        full_text = ss.read()
    print(Tokenizer.index_tokens(full_text))

