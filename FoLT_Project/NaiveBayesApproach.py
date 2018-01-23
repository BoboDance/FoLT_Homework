import logging
import random

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit


class NaiveBayesApproach:
    logging.basicConfig(level=logging.DEBUG)

    def __init__(self, train, dev):
        self.stop = stopwords.words('english')
        self.threshold = self.threshold_bigram = 1000

        self.train_data = train
        self.dev_data = dev

        self.word_cfd = None
        self.bigram_cfd = None

    def get_features(self, words, word_cfd, bigram_cfd):

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
        word_pos = sum([positive_fd[word] for word in words])
        word_neg = sum([-negative_fd[word] for word in words])
        res_dict.update({
            'pos_score': word_pos,
            'neg_score': word_neg,
            'difference': word_pos - word_neg,
            'is_positive': True if (word_pos - word_neg) > 0 else False
        })

        # words w/o stopwords, delimiters, etc.
        cleaned_words = [token.lower()
                         for token in words
                         if token not in self.stop
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

    def run(self):

        self.word_cfd = self.get_word_cfd()
        self.bigram_cfd = self.get_bigram_cfd()

        # train the model and check most informative features
        nbc = nltk.NaiveBayesClassifier.train(
            [(self.get_features(words, self.word_cfd, self.bigram_cfd), category) for
             (words, category) in
             self.train_data])
        print("Accuracy on test data:",
              nltk.classify.accuracy(nbc, [
                  (self.get_features(words, self.word_cfd, self.bigram_cfd), category) for
                  (words, category) in
                  self.dev_data]))
        print("\nMost informative features:")
        for elem in nbc.most_informative_features(20):
            print(elem)

    def get_word_cfd(self):
        fd_all_words = nltk.FreqDist([token.lower() for words, _ in self.train_data for token in words
                                      if token not in self.stop
                                      and token.isalnum()
                                      ])
        top_words = [word for (word, _) in fd_all_words.most_common(self.threshold)]

        logging.info("Word frequencies calculated and prepared.")

        return nltk.ConditionalFreqDist((category, token.lower())
                                        for words, category in self.train_data
                                        for token in words
                                        if token in top_words)

    def get_bigram_cfd(self):
        # get most frequent bigrams
        # basically the same as above, but with bigrams

        fd_all_bigrams = nltk.FreqDist(
            nltk.bigrams([token.lower() for words, _ in self.train_data for token in words
                          if token not in self.stop
                          and token.isalnum()
                          ]))
        top_bigrams = [bigram for (bigram, _) in fd_all_bigrams.most_common(self.threshold_bigram)]

        logging.info("Bigram frequencies calculated and prepared.")

        # get frequency of bigrams in train data
        return nltk.ConditionalFreqDist(
            (category, bigram)
            for words, category in self.train_data
            for bigram in nltk.bigrams([token.lower() for token in words])
            if bigram in top_bigrams)
