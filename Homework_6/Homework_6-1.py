import nltk
import re

from nltk.corpus import gutenberg, nps_chat, words, brown


def get_letters(digit):
    # convert digit to part of regex
    return {
        '1': '[\.,?]',
        '2': '[abc]',
        '3': '[def]',
        '4': '[ghi]',
        '5': '[jkl]',
        '6': '[mno]',
        '7': '[pqrs]',
        '8': '[tuv]',
        '9': '[wxyz]',
        '0': '[ ]',
    }[digit]


def get_T9_word(digits):

    # define start of regex sequence
    regex_search_string = "^"

    # append regex parts according to digits
    for digit in digits:
        regex_search_string += get_letters(digit)

    # add end of regex string
    regex_search_string += '$'

    # find all words, which fulfill regex pattern based in digit entry
    possible_words = [w for w in freq_words.keys() if re.search(regex_search_string, w)]

    # find best element by Freq in given corpus
    best_elem = ''
    best_freq = 0

    for word in possible_words:
        fw = freq_words[word]
        if fw > best_freq:
            best_elem = word
            best_freq = fw

    return best_elem


# This is bad for runtime if not defined as global variable :/
# This version is really bad from a performance perspective.
# You should consider in next years exercise to use the freqDist of words as param
# otherwise it is calculated every time the function is called, means for every word or is an ugly global variable.
freq_words = nltk.FreqDist(w.lower() for w in brown.words())

sentence = ['43556', '73837', '4', '26', '3463']
prediction = ''

for digits in sentence:
    prediction += get_T9_word(digits) + " "

print("Sequence ( 52473 ) is prediced as:", get_T9_word("252473"))
# remove last space
print("Sequence (", ", ".join(sentence), ") is predicted as: ", prediction[:-1])


# a)
# The first idea was to take the word corpus, because it contains most of the words given a language
# However, when thinking about it, it does not make sense.
# The corpus is basically a dictionary and contains a lot of words, but does not represent the
# freqDist of the given words.
# This means the first word which matches the regex pattern is chosen. ("gekko peter i am dime")
# Luckily 3 out of the 5 possible words were right.
# Therefore, a different corpus has to be used, ideally large one and with a lot of variety.
# The nps_chat corpus may come up, because SMS is basically chatting.
# Nonetheless, the results were not very pleasing: "hello  i am find"
# "Peter" was not even found in the corpus, this may be a result of its relatively small vocabulary (6066).
# Afterwards, the Gutenberg corpus seemed like a good choice, then was the way to good.
# It has a great range of words and additionally offers a large vocabulary (51156)
# to make freqDist more precise for English.
# Brown corpus is similar regarding the above features and just offers a slightly higher vocabulary size (56057).
# The final result as seen above uses brown for that reason.
# Also, the results of these two corpora do not vary at all: "hello peter i an find"

# c)
# The output produces a readable result, however it did not result in a syntactical correct English sentence.
# 3/5 words are classified correctly.
# The errors happen, because the distribution of "an" and "find" are higher.
# For "an" this is straight forward, as article, it is more widely used than "am",
# which can only be found in the context of "I".
# When thinking about it, "find" is reasonable as well, "find" can be used in many different situations.
# Whereas, "fine" is a bit more restricted to especially expressing feelings and as common answer in smalltalk/greeting.
# (How are you? Fine.)
