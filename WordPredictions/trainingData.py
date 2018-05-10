import numpy as np

def run(file):
    """
    takes in a file and opens it
    """
    f = open(file, "rt")
    text = f.read().lower()
    print(len(text))
    f.close()

    chars, char_indices, indices_char = uniqueChars(text)

    NUM_CHARS, sentences, next_chars = cut(text)

    x, Y = generateFeatures(chars, char_indices, NUM_CHARS, sentences, next_chars)

def uniqueChars(text):
    """
    takes a string and makes a list of unique characters, and two dictionary
    maps for characters: index and index: characters
    """
    # makes a list of every unique character in given text, sorted from a->z, 1->9, etc.
    chars = sorted( list(set(text)) )

    # creates a dictionary of each char in chars and its respective index in the chars list
    char_indices = dict((char, index) for index, char in enumerate(chars))

    #  creates a dictionary of each index in the chars list and the character in that index
    indices_char = dict((index, char) for index, char in enumerate(chars))

    print('Unique characters in text: ' + str(len(chars)))
    return (chars, char_indices, indices_char)


def cut(text):
    """
    Splits text into sequences of X chars, with each sequence starting 3 characters after the prior's start.
    TO SEE EXAMPLE-- code the following before return statement: print(sentences[500], sentences[501])
    """
    NUM_CHARS = 40      # number of characters in each sentence
    STEP = 3            # step forward by 3 characters
    sentences = []      # stores 40 char sentences to be used for training
    next_chars = []     # stores next character in each 40 character sentence

    for i in range(0, len(text) - NUM_CHARS, STEP):
        sentences += [ text[i : (i + NUM_CHARS)] ]
        next_chars += [ text[i + NUM_CHARS] ]

    print("Number of training examples: " + str(len(sentences)))
    return (NUM_CHARS, sentences, next_chars)


def generateFeatures(chars, char_indices, NUM_CHARS, sentences, next_chars):
    """
    """
    X = np.zeros((len(sentences), NUM_CHARS, len(chars)), dtype = np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for indexS, sent in enumerate(sentences):
        for indexC, char in enumerate(sent):
            X[indexS, indexC, char_indices[char]] = 1
        y[indexS, char_indices[next_chars[indexS]]] = 1
    return X, y


run("BeyondGoodandEvil.txt")