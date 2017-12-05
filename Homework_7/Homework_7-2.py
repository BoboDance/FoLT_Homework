import nltk
from nltk.corpus import brown
from texttable import Texttable


corpus = brown.words()
tagged_corpus_universal = nltk.pos_tag(corpus, tagset="universal")
tagged_corpus_universal_set = set(tagged_corpus_universal)

# create dict from keys [1, 10] in order to get at least 0
# when no word with that amount was found
count_dict = nltk.defaultdict(int).fromkeys(range(1, 11), 0)
for word, tags in nltk.Index(tagged_corpus_universal_set).items():
    # get amount of tags and ensure it is not more than 10, increment counter by one
    ctr = len(tags)
    if ctr > 10:
        continue
    count_dict[ctr] += 1

# this package makes printing tables easier
t = Texttable()
t.add_row(['Number of POS Tags', 'Number of Words'])
for k, v in count_dict.items():
    t.add_row([k, v])
print(t.draw())