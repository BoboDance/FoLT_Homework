import logging
import random

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit

from FoLT_Project.corpus_reviews import reviews

logging.basicConfig(level=logging.DEBUG)
random_seed = 0
stop = stopwords.words('english')


def get_features(fileid, word_cfd, bigram_cfd):
    text = reviews.words(fileid)

    res_dict = nltk.defaultdict(int)

    # score of resulting words
    # This basically takes the freq in the train set for pos and neg as score
    # Given this I have an automatic weighting:
    # In case a term does not only occur but also occurs often, it is rated higher
    # same for bigrams further down

    # This is a questionable feature.
    # The problem is: Both sentiments use similar words, even words like good.

    positive_fd = word_cfd["pos"]
    negative_fd = word_cfd["neg"]
    word_pos = sum([positive_fd[word] for word in text])
    word_neg = sum([-negative_fd[word] for word in text])
    res_dict.update({
        'pos_score': word_pos,
        'neg_score': word_neg,
        'difference': word_pos - word_neg,
        'is_positive': True if (word_pos - word_neg) > 0 else False
    })

    # words w/o stopwords, delimiters, etc.
    cleaned_words = [token.lower()
                     for token in text
                     if token not in stop
                     and token.isalnum()
                     ]

    # I also tried only using only the top words, no success
    text_fdist = nltk.FreqDist(cleaned_words)
    res_dict.update(text_fdist)

    # using sentiSynsets in order to determine positive or negative tendency
    # this is terrible: takes waaaaaaaaaaaaaaaaaaaay too long and is useless :(
    # res_pos_swn = sum([senti.pos_score() for word in text for senti in list(swn.senti_synsets(word))])
    # res_neg_swn = sum([senti.neg_score() for word in text for senti in list(swn.senti_synsets(word))])
    # my_dict.update({'senti_score_neg': res_neg_swn, 'senti_score_pos': res_pos_swn})

    bigrams = nltk.bigrams(cleaned_words)

    # just the bigram freqDist
    # I also tried only using the top bigrams, no success
    # I also tried just using uncleaned bigrams (included stopwords, delimiters, etc.), no changes in the results
    text_fdist_bigram = nltk.FreqDist(bigrams)
    res_dict.update(text_fdist_bigram)

    pos_score = 0
    neg_score = 0
    # get bigram score
    # basically checks if a bigram found in the text is more likely to be found in positive or negative reviews
    # the basis freqDist, which determines the freq of each bigram can be found below
    # Also important: remove stopwords otherwise this is stupid ("of the", "this is", etc.)
    for bigram in bigrams:
        neg_score += bigram_cfd['neg'][bigram]
        pos_score += bigram_cfd['pos'][bigram]
        # add score for each bigram, does not change anything
        # res_dict[str(bigram) + "_neg"] += bigram_cfd['neg'][bigram]
        # res_dict[str(bigram) + "_pos"] += bigram_cfd['neg'][bigram]

    res_dict.update({
        'pos_score_bi': pos_score,
        'neg score_bi': neg_score,
        'difference_bi': pos_score - neg_score,
        'is_positive_bi': True if pos_score - neg_score > 0 else False
    })

    # just some other plain simple features
    # this is more likely to decrease the performance than improve it
    # looks like there are not enough bad words in reviews :(
    # res_dict.update(
    #     {
    #         'length': len(text),
    #         "contains_bad_word": True if [word for word in text if word in badWords] else False
    #     })

    return res_dict


# prepare review data as a list of tuples (vocab, category)
review_data = [(fileid, category)
               for category in reviews.categories()
               for fileid in reviews.fileids(category)]

sss = StratifiedShuffleSplit(n_splits=3, random_state=random_seed, test_size=.2)

# X = [fileid for fileid, _ in review_data]
# y = [cat for _, cat in review_data]

# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

random.seed(random_seed)
random.shuffle(review_data)
logging.debug(str(review_data[:10]))

# split in training, develop and test set
train_size = int(0.8 * len(review_data))
train_data, dev_data = review_data[:train_size], review_data[train_size:]

print("Training data prepared.")

threshold = threshold_bigram = 1000

fd_all_words = nltk.FreqDist([w.lower() for fileid, _ in train_data for w in reviews.words(fileid)
                              if w not in stop
                              and w.isalnum()
                              ])
top_words = [word for (word, _) in fd_all_words.most_common(threshold)]

review_data_cfd = nltk.ConditionalFreqDist((category, token.lower())
                                           for fileid, category in train_data
                                           for token in reviews.words(fileid)
                                           if token in top_words)

print("Word frequencies calculated and prepared.")

# get most frequent bigrams
# basically the same as above, but with bigrams

fd_all_bigrams = nltk.FreqDist(nltk.bigrams([w.lower() for fileid, _ in train_data for w in reviews.words(fileid)
                                             if w not in stop
                                             and w.isalnum()
                                             ]))
top_bigrams = [bigram for (bigram, _) in fd_all_bigrams.most_common(threshold_bigram)]

# get frequency of bigrams in train data
review_data_cfd_bigrams = nltk.ConditionalFreqDist(
    (category, bigram)
    for fileid, category in train_data
    for bigram in nltk.bigrams([token.lower() for token in reviews.words(fileid)])
    if bigram in top_bigrams)

print("Bigram frequencies calculated and prepared.")

# train the model and check most informative features
nbc = nltk.NaiveBayesClassifier.train(
    [(get_features(fileid, review_data_cfd, review_data_cfd_bigrams), category) for (fileid, category) in
     train_data])
print("Accuracy on test data:",
      nltk.classify.accuracy(nbc, [(get_features(fileid, review_data_cfd, review_data_cfd_bigrams), category) for
                                   (fileid, category) in
                                   dev_data]))
print("\nMost informative features:")
for elem in nbc.most_informative_features(20):
    print(elem)
