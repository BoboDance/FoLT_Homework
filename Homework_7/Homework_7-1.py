import nltk
from nltk.corpus import brown
from collections import Counter


tagged_corpus = brown.tagged_words()
tagged_corpus_universal = brown.tagged_words(tagset="universal")
tagged_corpus_set = set(tagged_corpus)

# a)
# Capitalized words are intentionally included
# Daniil said we have to analyse them separately
# Anyway, here with only lower case words as well:
a = sorted(set([word.lower() for (word, tag) in tagged_corpus_set if tag == 'MD']))
# a = sorted([word for (word, tag) in tagged_corpus_set if tag == 'MD'])

# b)
# This list comprehension reduces the result
# to all singular third person verbs (VBZ)
# and all plural common nouns (NNS).
# Using a set for b avoids duplicates of "word : NNS/VBZ"
# This reduces the next step to finding duplicates.
b = [word for (word, tag) in tagged_corpus_set if tag == 'VBZ' or tag == 'NNS']

# return the duplicates, means matches of "word : [VBZ, NNS]"
b = [word for word, count in Counter(b).items() if count > 1]

# c)

# get indices of ADP elements
indices = [index for index, (_, tag) in enumerate(tagged_corpus_universal) if tag == 'ADP']

c = set()
for index in indices:
    # check if following two tokens are DET and NOUN
    det = tagged_corpus_universal[index+1]
    noun = tagged_corpus_universal[index+2]
    if det[1] != 'DET' or noun[1] != 'NOUN':
        continue
    # if yes add all three words to result
    c.add((tagged_corpus_universal[index][0], det[0], noun[0]))

# d)
feminine = {"she", "her", "herself", "hers"}
masculine = {"he", "his", "himself", "him"}

# pretty much just check if PRON matches on element in the above sets
# I have not found a matching POS tag for this
masculine_count = len([word for (word, tag) in tagged_corpus_universal if tag == "PRON" and word.lower() in feminine])
feminine_count = len([word for (word, tag) in tagged_corpus_universal if tag == "PRON" and word.lower() in masculine])

print("Distinct words with 'MD' POS tag in alphabetical order: ", a)
print("VBZ and NNS with same spelling: ", b)
print("ADP + DET + NN combinations : ", c)
print("Ratio of masculine to feminine pronouns: ", masculine_count / feminine_count)
