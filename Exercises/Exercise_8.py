# Task 8.1

# a) Everything as NN
# b) For the above sentence "only" is identified as ADV
# c) (r'^\.$', 'ADJ')
# d) *s --> VBZ; Names, like Molly --> ADV;

# Task 8.2

import nltk
from nltk.corpus import brown


def get_most_likely_tag(word, cfd):
    try:
        return cfd[word].max()
    except ValueError:
        return "UNK"


cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news", tagset="universal"))
print(get_most_likely_tag("lol", cfd))
for word in brown.sents(categories="science_fiction")[0]:
    print(word, get_most_likely_tag(word, cfd))

# Task 8.3

# a), b)

t0 = nltk.DefaultTagger('NOUN')
train_sents = brown.tagged_sents(tagset='universal', categories='news')
t1 = nltk.UnigramTagger(train_sents)
t2 = nltk.BigramTagger(train_sents)
t3 = nltk.TrigramTagger(train_sents)

sentences = [
    [("The", "DET"), ("only", "ADJ"), ("Conservative", "NOUN"), ("councillor", "NOUN"), ("representing", "VERB"),
     ("Cambridge", "NOUN"), ("resigned", "VERB"), ("from", "ADP"), ("the", "DET"), ("city", "NOUN"),
     ("council", "NOUN"), (".", ".")]]

print(t0.evaluate(sentences))
print(t1.evaluate(sentences))
print(t2.evaluate(sentences))
print(t3.evaluate(sentences))
