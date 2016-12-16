import word2vec
import numpy as np
import csv
import sys

related_word_pairs = [
    ("foot", "hand"),
    ("stroller", "car"),
    ("juic", "milk"),
    ("mouth", "nose"),
    ("book", "ball"),
    ("blanket", "diaper"),
    ("bottl", "spoon"),
    ("dog", "baby"),
    ("foot", "sock"),
    ("cooki", "banana")
]

unrelated_word_pairs = [
    ("sock", "juic"),
    ("milk", "foot"),
    ("banana", "nose"),
    ("mouth", "cooki"),
    ("blanket", "dog"),
    ("baby", "spoon"),
    ("book", "diaper"),
    ("juic", "car"),
    ("nose", "bottl"),
    ("hand", "stroller"),
    ("mouth", "ball")
]


def compare_words(model, a, b):
    """
    Returns cos(theta) of vector a and b.
    a and b are strings, which are keys to
    vectors held in the dictionary "model".

    model[a] and model[b] are already unit vectors
    """
    return np.dot(model[a], model[b])


def compare_wordpairs(model, wordpairs):
    """
    Calculates the cosine similarity of a
    list of tuples. Each tuple is a word pair.
    Returns a new list of tuples of the form:

    (word1, word2, cos(theta))

    """
    results = []
    for a, b in wordpairs:
        result = compare_words(model, a, b)
        results.append((a, b, result))
    return results

def average_cosine(cosines):
    return sum(result[2]
               for result in cosines)/len(cosines)

if __name__ == "__main__":

    input_corpus = sys.argv[1]
    dim = int(sys.argv[2])
    window = int(sys.argv[3])
    cbow = int(sys.argv[4])
    hs = int(sys.argv[5])



    word2vec.word2phrase(input_corpus, "out/phrases", verbose=True)

    word2vec.word2vec("out/phrases", "bin_out/childes.bin", size=dim, cbow=cbow, hs=hs, verbose=True)
