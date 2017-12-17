import nltk
from nltk.corpus import brown
from texttable import Texttable


def get_counts(tagged_corpus_universal_set):
    # container for best elems
    most_tagged_words = set()
    max_len = 1
    count_dict = nltk.defaultdict(int)

    for word, tags in nltk.Index(tagged_corpus_universal_set).items():
        # get amount of tags
        # save word to most_tagged if it has at least the same number of different tags as current most_tagged
        # increment counter by one for amount x of tags for one word
        ctr = len(tags)

        if ctr > max_len:
            max_len = ctr
            most_tagged_words = set()

        if ctr == max_len:
            most_tagged_words.add(word)

        count_dict[ctr] += 1

    return count_dict, max_len, most_tagged_words


def get_sents(tagged_corpus_universal, most_tagged_words, corpus_length, max_len):
    # get sents to most tagged words
    sentences = nltk.defaultdict(list)
    for word in most_tagged_words:
        checked_tags = set()

        # iterate over tagged corpus
        for i, item in enumerate(tagged_corpus_universal):

            # to lower is not necessary here,
            # because I already have a lowered corpus or I want to differentiate between caps and lower
            temp_word = item[0]
            tag = item[1]

            # check if word is in most_tagged collection
            if temp_word != word or tag in checked_tags:
                continue

            # temp container for one example sentence
            temp_sentence = ['\033[93m' + word + '\033[0m']

            # get word until previous punctuation
            iter_index = i - 1
            # this is a bit annoying, because '.' tag also contains commas, therefore you do not get the whole sentence.
            while not (tagged_corpus_universal[iter_index][1] == "."
                       and tagged_corpus_universal[iter_index][0] in ('.', '!', '?')):
                # append before the most tagged word
                temp_sentence = [tagged_corpus_universal[iter_index][0]] + temp_sentence
                iter_index -= 1
                # avoid problems with begin of corpus
                if iter_index < 0:
                    break

            # get word until next punctuation
            # same problem here
            iter_index = i + 1
            while not (tagged_corpus_universal[iter_index][1] == "."
                       and tagged_corpus_universal[iter_index][0] in ('.', '!', '?')):
                # append after word
                temp_sentence.append(tagged_corpus_universal[iter_index][0])
                iter_index += 1
                # avoid problems with missing '.' tag at the end of corpus
                if iter_index > corpus_length:
                    break

            # add temp sent to list of all examples for this word.
            # add to checked_tags to avoid finding an example with same tag again.
            sentences[word].append(tag + ": " + " ".join(temp_sentence) + ".")
            checked_tags.add(tag)

            # when examples for all tags found, start with next word
            if len(checked_tags) == max_len:
                break

    return sentences


# using lower case will result in a different result:
# 'down' then also has a ADV tag
# for comparison I printed both

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Not Lower
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Results for not using lower case: \n")

tagged_corpus_universal_normal = brown.tagged_words(tagset="universal")
length_normal = len(tagged_corpus_universal_normal)
tagged_corpus_universal_set_normal = set(tagged_corpus_universal_normal)
counts, length_max, most_tags = get_counts(tagged_corpus_universal_set_normal)

# this package makes printing tables easier (pip install texttable)
t = Texttable()
t.header(['Number of POS Tags', 'Number of Words'])
# iterate over keys [1, 10] in order to get at least 0
# when no word with that amount of tags was found
for i in range(1, 11):
    t.add_row([i, counts[i]])
print(t.draw() + "\n")

sents = get_sents(tagged_corpus_universal=tagged_corpus_universal_normal,
                  most_tagged_words=most_tags, corpus_length=length_normal,
                  max_len=length_max)

print("Usage of words with {} tags w/o lower case.".format(length_max))
t = Texttable()
t.header(['Word', 'Sentences'])
t.set_cols_width([10, 500])
# iterate over keys [1, 10] in order to get at least 0
# when no word with that amount of tags was found
for word, example in sents.items():
    t.add_row(["\n" * (length_max // 2) + word, "\n".join(example)])
print(t.draw() + "\n")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Lower
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Results for using lower case: \n")

tagged_corpus_universal_lower = [(word.lower(), tag) for (word, tag) in brown.tagged_words(tagset="universal")]
length_lower = len(tagged_corpus_universal_lower)
tagged_corpus_universal_set_lower = set(tagged_corpus_universal_lower)

counts, length_max, most_tags = get_counts(tagged_corpus_universal_set_lower)

# this package makes printing tables easier (pip install texttable)
t = Texttable()
t.header(['Number of POS Tags', 'Number of Words'])
# iterate over keys [1, 10] in order to get at least 0
# when no word with that amount of tags was found
for i in range(1, 11):
    t.add_row([i, counts[i]])
print(t.draw() + "\n")

sents = get_sents(tagged_corpus_universal=tagged_corpus_universal_lower,
                  most_tagged_words=most_tags, corpus_length=length_lower,
                  max_len=length_max)

print("Usage of words with {} tags w/ lower case".format(length_max))
t = Texttable()
t.header(['Word', 'Sentences'])
t.set_cols_width([10, 500])
# iterate over keys [1, 10] in order to get at least 0
# when no word with that amount of tags was found
for word, example in sents.items():
    t.add_row(["\n" * (length_max // 2) + word, "\n".join(example)])
print(t.draw() + "\n")
