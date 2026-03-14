import re
from collections import Counter


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\säöüß]', '', text)
    return text.split()


def build_vocab(texts, min_freq=2, max_size=10000):
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    # Keep only words above min frequency and up to max_size
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_text(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def pad_sequence(seq, max_len, pad_value=0):
    return seq + [pad_value]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
