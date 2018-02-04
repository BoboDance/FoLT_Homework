import logging
import random
import zipfile

from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords


class DataTransformation:
    """
    Using numpy arrays for the split is somehow really bad and takes much longer, therefore I use simple lists
    This is mainly a class for printing, reading and transforming data as necessary.
    """

    def __init__(self, data_resource, data_resource_test, is_real_test=False):
        self.random_seed = 0
        self.data_resource = data_resource
        self.data_resource_test = data_resource_test
        self.stop_words = stopwords.words("english")
        self.is_real_test = is_real_test

    def split_data(self):
        # prepare review data as a list of tuples (vocab, category)
        review_data = [(fileid, category)
                       for category in self.data_resource.categories()
                       for fileid in self.data_resource.fileids(category)]

        random.seed(self.random_seed)
        random.shuffle(review_data)

        if self.is_real_test:
            train_data = review_data
            test_data = [fileid
                         for fileid in self.data_resource_test.fileids()]
        else:
            # split in training, develop and test set
            train_size = int(0.8 * len(review_data))
            train_data, test_data = review_data[:train_size], review_data[train_size:]

        logging.info("Training data prepared.")

        return train_data, test_data

    def get_raw_with_split(self, data):
        x = []
        y = []
        for fileid, category in data:
            x.append(self.get_raw(fileid))
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

    def get_test_data(self, data, is_get_raw):
        if is_get_raw:
            xy = [(fileid, self.data_resource_test.raw(fileid)) for fileid in data]
        else:
            xy = [(fileid, self.data_resource_test.words(fileid)) for fileid in data]

        logging.info("Returned test words/raw.")
        return dict(xy)

    def get_word(self, fileid):
        return self.data_resource.words(fileid)

    def get_sent(self, fileid):
        return self.data_resource.sents(fileid)

    def get_raw(self, fileid):
        return self.data_resource.raw(fileid)

    def transform_sents(self, docs, labels, is_train):
        """Transform sents for doc2vec"""
        neg_ctr = 0
        pos_ctr = 0
        pos = []
        neg = []
        prefix = "train" if is_train else "test"
        for i, sents in enumerate(docs):
            sent_class = labels[i]

            if sent_class == "neg":
                neg.append(TaggedDocument(self.remove_stopwords(sents),
                                          ['{}_{}_{}'.format(prefix, sent_class, neg_ctr)]))

                neg_ctr += 1
            else:
                pos.append(TaggedDocument(self.remove_stopwords(sents),
                                          ['{}_{}_{}'.format(prefix, sent_class, pos_ctr)]))
                pos_ctr += 1

        logging.info("Finished formatting documents.")
        return pos, neg

    @staticmethod
    def create_classifier_arrays(doc2vec_model, is_train, pos_data_length, neg_data_length):
        """Create train/test arrays from doc2vec model"""
        arrays = []
        labels = []
        prefix = "train" if is_train else "test"

        for i in range(pos_data_length):
            arrays.append(doc2vec_model.docvecs['{}_pos_{}'.format(prefix, str(i))])
            labels.append("pos")

        for i in range(neg_data_length):
            arrays.append(doc2vec_model.docvecs['{}_neg_{}'.format(prefix, str(i))])
            labels.append("neg")

        return arrays, labels

    def remove_stopwords_corpus(self, data):
        return [
            [word.lower()
             for word in text
             if word not in self.stop_words
             ]
            for text in data
        ]

    def remove_stopwords(self, tokens):
        return [word.lower()
                for word in tokens
                if word not in self.stop_words]

    @staticmethod
    def write_to_file(result_dict, name):
        """create submission zip"""
        f = open("answer.txt", mode="w")
        for item in ["%s_%s" % (key[:-4], value) for key, value in result_dict.items()]:
            f.write("%s\n" % item)
        f.close()
        z = zipfile.ZipFile('results-{}.zip'.format(name), mode='w')
        z.write("answer.txt")
        z.close()

