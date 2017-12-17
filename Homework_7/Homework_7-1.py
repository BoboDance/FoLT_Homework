import nltk
from nltk.corpus import brown
from collections import Counter


tagged_corpus = brown.tagged_words()
tagged_corpus_universal = brown.tagged_words(tagset="universal")
tagged_corpus_set = set(tagged_corpus)

# a)
# Capitalized words are intentionally included
# Daniil said we have to analyse them separately.
# a = sorted([word for (word, tag) in tagged_corpus_set if tag == 'MD'])
# Anyway, in the forum there was a question and it said to lower all words, here with only lower case words as well:
# therefore I have done this for every exercise, where it makes sense.
a = sorted(set([word.lower() for (word, tag) in tagged_corpus_set if tag == 'MD']))

# b)
# This list comprehension reduces the result
# to all singular third person verbs (VBZ),
# and all plural common nouns (NNS) an proper nouns (NNPS).
# These two both end in -s and may result in an ambiguity.
# Using a set for b avoids duplicates of "word : NNS/VBZ"
# This reduces the next step to finding duplicates.
# make set to avoid duplicates, i think only distinct values are interesting, not the same match over and over again.
vbz = set([word.lower() for (word, tag) in tagged_corpus_set if tag == 'VBZ'])
nn = set([word.lower() for (word, tag) in tagged_corpus_set if tag in ('NNS', 'NNPS')])

# return the duplicates, means matches of "word : [VBZ, NNS/NNPS]"
b = vbz.intersection(nn)

# c)
# get indices of ADP elements
indices = [index for index, (_, tag) in enumerate(tagged_corpus_universal) if tag == 'ADP']

c = set()
for index in indices:
    # check if the following two tokens are DET and NOUN
    det = tagged_corpus_universal[index+1]
    noun = tagged_corpus_universal[index+2]
    if det[1] != 'DET' or noun[1] != 'NOUN':
        continue
    # if yes, add pattern of three words to result
    # to lower to remove duplicates if ADP is e.g. found at the beginning
    c.add((tagged_corpus_universal[index][0].lower(), det[0].lower(), noun[0].lower()))

# d)
# define f and m pronouns, because treebank does not offer a tag for that.
feminine = {"she", "her", "herself", "hers"}
masculine = {"he", "his", "himself", "him"}

# this pretty much just checks, if PRON matches one of the elements in the above sets
# In this case lower() does not matter, because both times it is counted as 1.
masculine_count = len([word for (word, tag) in tagged_corpus_universal if tag == "PRON" and word.lower() in feminine])
feminine_count = len([word for (word, tag) in tagged_corpus_universal if tag == "PRON" and word.lower() in masculine])

print("Distinct words with 'MD' POS tag in alphabetical order: ", a)
print("VBZ and NNS with same spelling: ", b)
print("ADP + DET + NN combinations : ", c)
print("Ratio of masculine to feminine pronouns: ", masculine_count / feminine_count)
