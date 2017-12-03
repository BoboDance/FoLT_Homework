import math
import nltk
import collections
from nltk.corpus import brown, gutenberg, reuters


# ----------------------------------------
# Helper functions
# ----------------------------------------

def find_suitable_text():
    """Find suitable text for background with checking length of Gutenberg texts,
    Brown categories and Reuters categories. Texts or categories over 50k words are marked green.
    Total length of the corpus is at the top as header"""
    print('\033[95m')
    print("--------------- Gutenberg ---------------")
    print("Total Length: ", len(gutenberg.words()))
    print('\033[0m')
    for fid in gutenberg.fileids():
        words = gutenberg.words(fid)
        length = len(words)
        if length > 50000:
            print('\033[92m')
        print("Text: ", fid)
        print("Length: ", length)
        print("Content preview: ", words[:20])
        if length > 50000:
            print('\033[0m')
        else:
            print("")

    # brown texts are too short, therefore check categories
    print('\033[95m')
    print("--------------- Brown ---------------")
    print("Total Length: ", len(brown.words()))
    print('\033[0m')
    for cat in brown.categories():
        words = brown.words(categories=cat)
        length = len(words)
        if length > 50000:
            print('\033[92m')
        print("Text category: ", cat)
        print("Length: ", length)
        print("Content preview: ", words[:20])
        if length > 50000:
            print('\033[0m')
        else:
            print("")

    # reuters texts are too short, therefore check categories
    # reuters actually has some funny categories
    # reuters categories are rather small, however the total corpus is quire large
    print('\033[95m')
    print("--------------- Reuters ---------------")
    print("Total Length: ", len(reuters.words()))
    print('\033[0m')
    for cat in reuters.categories():
        words = reuters.words(categories=[cat])
        if length > 50000:
            print('\033[92m')
        print("Text category: ", cat)
        print("Length: ", len(words))
        print("Content preview: ", words[:20])
        if length > 50000:
            print('\033[0m')
        else:
            print("")

def compute_LL(phrase, fdist_fg, fdist_bg):
    a = fdist_fg[phrase]
    b = fdist_bg[phrase]

    c = fdist_fg.N()
    d = fdist_bg.N()

    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)

    # avoid issues with log(0)
    helper = 0
    if b != 0:
        helper = b * math.log2(b / e2)

    return 2 * (a * math.log2(a / e1) + helper)

# ----------------------------------------
# Execute code
# ----------------------------------------

# This is only needed once to find a text, it can be uncommented if necessary
# find_suitable_text()

# whole brown corpus offers a greater variety of text styles and therefore common phrases
# this makes it a good choice as corpus to represent the distribution of phrases.
brown_bigram = nltk.bigrams(w.lower() for w in brown.words()) # generator
fdist_bg = nltk.FreqDist(brown_bigram)

# moby dick for fg, because why not
input_bigram = nltk.bigrams(w.lower() for w in gutenberg.words("melville-moby_dick.txt"))# generator
fdist_fg = nltk.FreqDist(input_bigram)

result = nltk.defaultdict(int)

# add LL values to defaultdict
for elem in fdist_fg.keys():
    result[elem] = compute_LL(phrase=elem, fdist_fg=fdist_fg, fdist_bg=fdist_bg)

# find the top 10 most common bigrams regarding LL
for k, v in collections.Counter(result).most_common(10):
    print(k, v)
