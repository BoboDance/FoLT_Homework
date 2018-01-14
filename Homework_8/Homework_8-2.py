import nltk
from nltk.corpus import nps_chat, brown
from nltk.tag import hmm


def tag_it(train, test, regex_pattern, print_errors=False):
    """
    Use tagger hierarchy approach shown in the lecture
    I actually tried some variations and different orders, e.g. regex at the beginning.
    But the below order gave me the best results
    :param train:
    :param test:
    :param regex_pattern:
    :param print_errors:
    :return:
    """

    default_tagger = nltk.DefaultTagger('NOUN')
    regex_tagger = nltk.tag.RegexpTagger(regex_pattern, backoff=default_tagger)
    unigram_tagger = nltk.UnigramTagger(train, backoff=regex_tagger)
    bigram_tagger = nltk.BigramTagger(train, backoff=unigram_tagger)
    trigram_tagger = nltk.TrigramTagger(train, backoff=bigram_tagger)

    print(trigram_tagger.evaluate(test))

    # print wrongly classified values
    if print_errors:
        sents = nps_chat.posts()
        untagged = trigram_tagger.tag_sents(sents[((len(sents) * 9) // 10):])
        cfd = nltk.ConditionalFreqDist((word, tag)
                                       for idx1, sent in enumerate(test)
                                       for idx2, (word, tag) in enumerate(sent)
                                       if tag != untagged[idx1][idx2][1]
                                       )

        for k, v in cfd.items():
            for key, item in v.items():
                print(k, key, item)


def hmm_tagging(train, test):
    """
    Train a HMM for prediction.
    Did not work so well.
    Especially when not using the eplacement method for really rare words. (Freq <=2)
    :param train: Annotated training data
    :param test: Annotated test data for eval
    :return:
    """
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train)
    print(tagger.evaluate(test))


def replace_rare(freqDist, words, n):
    """
    Replace words in corpus with UNK if they occur <= n
    :param freqDist: Frequencies of words in training corpus
    :param words: tagged sents
    :param n: limit for elimination
    :return: new list of sents with replace words
    """
    res_words = []
    for idx1, sent in enumerate(words):
        res_sent = []
        for idx2, (word, tag) in enumerate(sent):
            res_word = (word, tag)
            if freqDist[word] <= n:
                res_word = ('UNK', tag)
            res_sent.append(res_word)
        res_words.append(res_sent)

    return res_words


# chat train, dev and test
tagged_posts = nps_chat.tagged_posts(tagset='universal')
nr_posts = len(tagged_posts)
train_posts = tagged_posts[:(nr_posts * 9) // 10]
dev_posts = tagged_posts[(nr_posts * 8) // 10:(nr_posts * 9) // 10]
test_posts = tagged_posts[((nr_posts * 9) // 10):]

# some default regex patterns, plus some chat specific stuff
word_patterns = [
    (r'.*ould$', 'VERB'),
    (r'.*ing$', 'VERB'),
    (r'.*ed$', 'VERB'),
    (r'.*es$', 'VERB'),
    (r'.*\'s$', 'NOUN'),
    (r'.*s\'$', 'NOUN'),
    (r'.*ness$', 'NOUN'),
    (r'.*ment$', 'NOUN'),
    (r'.*ful$', 'ADJ'),
    (r'.*ious$', 'ADJ'),
    (r'.*ble$', 'ADJ'),
    (r'.*ic$', 'ADJ'),
    (r'.*ive$', 'ADJ'),
    (r'.*ic$', 'ADJ'),
    (r'.*est$', 'ADJ'),
    (r'.*ly$', 'ADV'),
    (r'^a$', 'DET'),
    (r'.*(.)\1{2}.*', 'X'),  # one char is repeated more than twice e.g. matches: niiiiiiice; does not: math running
    (r'.*[\.,!\?]+.*', '.'),  # if punctuation is included assign '.' tag,
    # this actually improves the result more than expected, probably related to chat things like "yes!!!!!!!"
    (r'^-?[0-9]+(.[0-9]+)?\%?$', 'NUM'),  # numbers and percentages
    (r'.*in\'?$', 'VERB'),  # slang pattern for e.g. tellin or tellin'
    (r'[A-Z].*', 'NOUN'),  # Noun if first letter is capital case
]

train_corp_posts = train_posts + dev_posts

#############################################################################################
# Remarks
# All dev tests are excluded here and were run previously
# In order to make all the approaches comparable, all are run against the test set.
#############################################################################################

# suggestions from the lecture:
# replace rare or unknown words with UNK in order to improve performance
# especially for bigramm and trigram tagger
# I thought for a corpus like the chat corpus, with a lot of different unknown words (niiiiice, etc.),
# this would yield in better results.
# However, the results did not improve compared to the plain, straight forward approach.
freqDist_train = nltk.FreqDist(word
                               for sent in train_corp_posts for (word, _) in sent
                               )
train_unk = replace_rare(freqDist_train, train_corp_posts, 2)
test_unk = replace_rare(freqDist_train, test_posts, 0)
print("Result w/ replacement of rare words (just chat training):")
tag_it(train_unk, test_unk, word_patterns)
print("Result w/o replacement of rare words (just chat training):")
tag_it(train_corp_posts, test_posts, word_patterns)

#############################################################################################

# Choosing brown as additional training corpus did not help either.
# I think, because of the great difference in language, the tags where predicted wrong.
# this is uncommented, otherwise the script takes quite some time.

# brown train and dev
brown_tagged = brown.tagged_sents(tagset='universal')
nr_posts = len(brown_tagged)
train_brown = brown_tagged[:(nr_posts * 9) // 10]
dev_brown = brown_tagged[((nr_posts * 9) // 10):]

train_corp_brown = train_brown + dev_brown
train_all = train_corp_brown + train_corp_posts

freqDist_train = nltk.FreqDist(word
                               for sent in train_all for (word, _) in sent
                               )
train_unk = replace_rare(freqDist_train, train_all, 2)
test_unk = replace_rare(freqDist_train, test_posts, 0)
print("Result w/ replacement of rare words (brown and chat training):")
tag_it(train_unk, test_unk, word_patterns)
print("Result w/o replacement of rare words (brown and chat training):")
tag_it(train_all, test_posts, word_patterns)

#############################################################################################

# HMM approach

# From intuition I thought this might perform best.
# But I think this approach needs more training data (in the right genre)
# in order to calculate more reliable probabilities and predict the tags correctly.
# But I still get a decent score for accuracy with this one, nonetheless not as good as the simple approach.

freqDist_train = nltk.FreqDist(word
                               for sent in train_corp_posts for (word, _) in sent
                               )
train_unk = replace_rare(freqDist_train, train_corp_posts, 2)
test_unk = replace_rare(freqDist_train, test_posts, 0)
print("Result HMM w/ replacement of rare words (just chat training):")
hmm_tagging(train_unk, test_unk)
print("Result HMM w/o replacement of rare words (just chat training):")
hmm_tagging(train_corp_posts, test_posts)

freqDist_train = nltk.FreqDist(word
                               for sent in train_all for (word, _) in sent
                               )
train_unk = replace_rare(freqDist_train, train_all, 2)
test_unk = replace_rare(freqDist_train, test_posts, 0)
print("Result HMM w/ replacement of rare words (brown and chat training):")
hmm_tagging(train_unk, test_unk)
print("Result HMM w/o replacement of rare words (brown and chat training):")
hmm_tagging(train_all, test_posts)
