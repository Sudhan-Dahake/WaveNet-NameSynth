from dependencies import *
from Variables import *

class Vocabulary:
    def __init__(self, words):
        self.words = words
        chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s: i+1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.itos)


    def __call__(self, words):
        X, Y = [], []

        for w in words:
            context = block_size * [0]
            for char in w + '.':
                ix = self.stoi[char]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y