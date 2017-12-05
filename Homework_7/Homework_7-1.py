import nltk
from nltk.corpus import brown
from collections import Counter


corpus = brown.words()
tagged_corpus = nltk.pos_tag(corpus)
tagged_corpus_universal = nltk.pos_tag(corpus, tagset="universal")
tagged_corpus_set = set(tagged_corpus)

# a)
a = [word for (word, tag) in tagged_corpus_set if tag == 'MD']
a = sorted(a)

# b)
# This list comprehension reduces the result
# to all singular third person verbs (VBZ)
# and all plural common nouns (NNS), which end in -s/-es
b = [word for (word, tag) in tagged_corpus_set if tag == 'VBZ' or tag == 'NNS']

# get duplicates, means matches of "word : [VBZ, NNS]"
# Using a set for b avoids matches like "word : [NNS, NNS]", etc.
b = [word for word, count in Counter(b).items() if count > 1]

# c)

# get indices of ADP elements
indices = [index for index, (_, tag) in enumerate(tagged_corpus_universal) if tag == 'ADP']

c = set()
for index in indices:
    det = tagged_corpus_universal[index+1]
    noun = tagged_corpus_universal[index+2]
    if det[1] != 'DET' or noun[1] != 'NOUN':
        continue
    c.add((tagged_corpus_universal[index][0], det[0], noun[0]))

# d)
# The Brown corpus only provides English text.
# This means Pronouns do not have gender, unlike German, French, Spanish, etc.
# Therefore, the ratio is 0, 1 or undefined.
# Depending on your point of view.

print("Distinct words with 'MD' POS tag in alphabetical order: ", a)
print("VBZ and NNS with same spelling: ", b)
print("ADP + DET + NN combinations : ", c)
