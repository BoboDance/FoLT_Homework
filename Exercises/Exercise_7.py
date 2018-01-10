import nltk
from nltk.corpus import brown

# 7.1

sent = "Racing down the hill on skis is fun, the race earlier was a great experience, while I ski."
print(nltk.pos_tag(nltk.word_tokenize(sent)))

# other options:
# saw
# dance
# smoke

# 7.2


def find_homonym(text):
    pos = nltk.pos_tag(text)
    pos_set = set(pos)
    if len(pos_set) == len(pos):
        return

    print([(k,v) for k,v in nltk.Index(pos_set).items() if len(v) > 1])


find_homonym(brown.words())