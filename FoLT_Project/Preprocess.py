import logging
import random

import numpy as np


class Preprocess:

    """
    Using numpy arrays for the split is somehow really bad and takes much longer, therefore I use simple lists
    """

    def __init__(self, data_resource):
        self.random_seed = 0
        self.data_resource = data_resource

    def split_data(self):
        # prepare review data as a list of tuples (vocab, category)
        review_data = [(fileid, category)
                       for category in self.data_resource.categories()
                       for fileid in self.data_resource.fileids(category)]

        # sss = StratifiedShuffleSplit(n_splits=3, random_state=self.random_seed, test_size=.2)

        # X = [fileid for fileid, _ in review_data]
        # y = [cat for _, cat in review_data]

        # for train_index, test_index in sss.split(X, y):
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]

        random.seed(self.random_seed)
        random.shuffle(review_data)
        # logging.debug(str(review_data[:10]))

        # split in training, develop and test set
        train_size = int(0.8 * len(review_data))
        train_data, dev_data = review_data[:train_size], review_data[train_size:]

        logging.info("Training data prepared.")

        return train_data, dev_data

    def get_raw_with_split(self, data):
        x = []
        y = []
        for fileid, category in data:
            x .append(self.get_raw(fileid))
            y.append(category)

        logging.info("Returned raw data with split.")
        return x, y

    def get_words_without_split(self, data):
        xy = [(self.get_word(fileid), category) for fileid, category in data]
        logging.info("Returned words without split.")
        return xy

    def get_words_with_split(self, data):
        x = []
        y = []
        for fileid, category in data:
            x.append(self.get_word(fileid))
            y.append(category)

        logging.info("Returned words data with split.")
        return x, y

    def get_word(self, fileid):
        return self.data_resource.words(fileid)

    def get_sent(self, fileid):
        return self.data_resource.sents(fileid)

    def get_raw(self, fileid):
        return self.data_resource.raw(fileid)
