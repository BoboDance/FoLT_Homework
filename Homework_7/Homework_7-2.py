import nltk
from nltk.corpus import brown
from texttable import Texttable


tagged_corpus_universal = brown.tagged_words(tagset="universal")
corpus_length = len(tagged_corpus_universal)
tagged_corpus_universal_set = set(tagged_corpus_universal)

# container for best elems
most_tagged_words = set()
max_len = 1
count_dict = nltk.defaultdict(int)

for word, tags in nltk.Index(tagged_corpus_universal_set).items():
    # get amount of tags increment counter by one
    # save word to most_tagged if
    ctr = len(tags)

    if ctr > max_len:
        max_len = ctr
        most_tagged_words = set()

    if ctr == max_len:
        most_tagged_words.add(word)

    count_dict[ctr] += 1

# this package makes printing tables easier
t = Texttable()
t.header(['Number of POS Tags', 'Number of Words'])
# iterate over keys [1, 10] in order to get at least 0
# when no word with that amount of tags was found
for i in range(1, 11):
    t.add_row([i, count_dict[i]])
print(t.draw() + "\n")

sentences = nltk.defaultdict(list)
for word in most_tagged_words:
    checked_tags = set()

    for i, item in enumerate(tagged_corpus_universal):

        temp_word = item[0]
        tag = item[1]

        if temp_word != word or tag in checked_tags:
            continue

        temp_sentence = [word]

        # get word until previous punctuation
        iter_index = i-1
        while not (tagged_corpus_universal[iter_index][1] == "." and tagged_corpus_universal[iter_index][0] == '.'):
            temp_sentence = [tagged_corpus_universal[iter_index][0]] + temp_sentence
            iter_index -= 1
            if iter_index < 0:
                break

        # get word until next punctuation
        iter_index = i+1
        while not (tagged_corpus_universal[iter_index][1] == "." and tagged_corpus_universal[iter_index][0] == '.'):
            temp_sentence.append(tagged_corpus_universal[iter_index][0])
            iter_index += 1
            if iter_index > corpus_length:
                break

        sentences[word].append(tag + ": " + " ".join(temp_sentence) + ".")
        checked_tags.add(tag)

        if len(checked_tags) == max_len:
            break

print("Usage of words with most tags of ", max_len)
t = Texttable()
t.header(['Word', 'Sentences'])
t.set_cols_width([10, 500])
# iterate over keys [1, 10] in order to get at least 0
# when no word with that amount of tags was found
for word, sents in sentences.items():
    t.add_row([word + "\n"*max_len, "\n".join(sents)])
print(t.draw() + "\n")
