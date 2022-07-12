import nltk
from collections import defaultdict
import numpy as np


def ngrams(sent, n=2, pad_left=False, pad_right=False):
    sent = [None]*(n-1)*pad_left + sent + [None]*(n-1)*pad_right
    return (tuple(sent[i:i+n]) for i in range(len(sent) - n + 1))

def make_ngrammodel(corp, f, n=2):
    """makes an ngrammodel where n >= 2. Takes a corpus corp from nltk, and a file name f from the corpus."""
    sents = corp.sents(f)

    ngram_count = defaultdict(lambda: defaultdict(lambda: 0))
    ngrammodel = defaultdict(lambda: defaultdict(lambda: 0.0))
    
    # w = word
    for sent in sents:
        for ws in ngrams(sent, n=n, pad_left=True, pad_right=True):
            wp, wn = ws[:n-1], ws[-1]
            ngram_count[wp][wn] += 1

    for wp in ngram_count:
        sum_ngram_count = sum(ngram_count[wp].values())
        for wn in ngram_count[wp]:
            ngrammodel[wp][wn] = ngram_count[wp][wn] / sum_ngram_count
    return ngrammodel

def ngram_gen_sent(model, n=2):
    """Generates a sentence from a ngram model."""
    sent = [None]*(n-1)
    sent_prob = 1
    while True:
        key = tuple(sent[-(n-1):])
        wrds = list(model[key].keys())
        prob = list(model[key].values())
        #print(model)
        sent.append(np.random.choice(wrds, p=prob))
        sent_prob *= model[sent[-2]][sent[-1]]
        if not sent[-1]:
            break
    return sent, sent_prob

def __main__():
    n = 2
    bm = make_ngrammodel(nltk.corpus.gutenberg, "bible-kjv.txt", n=n)
    sent, sent_prob = ngram_gen_sent(bm, n=n)
    print(f"Prob     | %.50f" % sent_prob)
    print(f"Sentence | {' '.join(w for w in sent if w)}")

if __name__ == "__main__":
    __main__()
