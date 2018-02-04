import logging
import os
import random

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from gensim import corpora, models, similarities
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC


from FoLT_Project.BaseApproach import BaseApproach
from FoLT_Project.Visualization import Visualization

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_location = os.path.dirname(os.path.realpath(__file__)) + "/models"


class Paragraph2Vec(BaseApproach):

    def __init__(self, train_x, train_y, test_x, test_y, is_real_test, data_transformation, build_model=False):

        super().__init__()
        self.data_transformation = data_transformation
        self.is_real_test = is_real_test
        self.build_model = build_model

        self.iter = 100

        self.train_x = train_x
        self.train_y = train_y

        self.test_x = test_x
        # just set all labels pos if not provided
        self.test_y = test_y if not is_real_test else ["pos" for _ in range(0, len(test_x))]

        self.corpus = None
        self.dictionary = None
        self.tfidf_model = None
        self.lsi_model = None
        self.index = None

        self.doc2vec_model = None
        self.doc2vec_model_name = '/reviews.doc2vec'  # '/reviews_dev.doc2vec'
        self.tagged_docs_pos_train = None
        self.tagged_docs_neg_train = None
        self.tagged_docs_pos_test = None
        self.tagged_docs_neg_test = None

    def run(self):

        # self.calc_similarity()
        self.get_doc2vec_model(self.build_model)

    def calc_similarity(self):
        self.get_models()

        texts_dev = self.data_transformation.remove_stopwords_corpus(self.test_x)

        # for sent in texts_dev:
        vec_bow = self.dictionary.doc2bow(texts_dev[0])
        vec_lsi = self.lsi_model[vec_bow]
        similar = self.index[vec_lsi]
        print(zip(similar, self.test_y))
        similar = sorted(enumerate(similar), key=lambda item: -item[1])
        print(similar)

    def get_models(self):
        if self.build_model:
            texts_train = self.data_transformation.remove_stopwords_corpus(self.train_x)
            dictionary = corpora.Dictionary(texts_train)
            dictionary.save(model_location + '/reviews.dict')

            self.corpus = [dictionary.doc2bow(text) for text in texts_train]
            corpora.MmCorpus.serialize(model_location + '/reviews.mm', self.corpus)

            self.tfidf_model = models.TfidfModel(self.corpus)
            self.tfidf_model.save(model_location + '/reviews.tfidf')

            # initialize an LSI transformation and
            # create wrapper over the original corpus: bow->tfidf->fold-in-lsi
            corpus_tfidf = self.tfidf_model[self.corpus]
            self.lsi_model = models.LsiModel(corpus_tfidf,
                                             id2word=self.dictionary, num_topics=300)
            self.lsi_model.save(model_location + '/reviews.lsi')

            # index corpus
            self.index = similarities.MatrixSimilarity(self.lsi_model[self.corpus])
            self.index.save(model_location + '/reviews.index')

        elif os.path.exists(model_location + '/reviews.dict') \
                and os.path.exists(model_location + '/reviews.mm') \
                and os.path.exists(model_location + '/reviews.tfidf') \
                and os.path.exists(model_location + '/reviews.lsi') \
                and os.path.exists(model_location + '/reviews.index'):

            self.dictionary = corpora.Dictionary()
            self.dictionary.load(model_location + '/reviews.dict')

            self.corpus = corpora.MmCorpus(model_location + '/reviews.mm')
            self.tfidf_model = models.TfidfModel.load(model_location + '/reviews.tfidf')
            self.lsi_model = models.LsiModel.load(model_location + '/reviews.lsi')
            self.index = similarities.MatrixSimilarity.load(model_location + '/reviews.index')

        else:
            raise ValueError("You have to build a model dict and corpus first.")

    def get_doc2vec_model(self, build_model):

        self.tagged_docs_pos_train, self.tagged_docs_neg_train = self.data_transformation.transform_sents(
            self.train_x,
            self.train_y,
            True)

        x = self.test_x if not self.is_real_test else self.test_x.values()
        self.tagged_docs_pos_test, self.tagged_docs_neg_test = self.data_transformation.transform_sents(x,
                                                                                                        self.test_y,
                                                                                                        False)

        if build_model or not os.path.exists(model_location + self.doc2vec_model_name):

            self.doc2vec_model = models.Doc2Vec(min_count=1, window=10, size=400, sample=1e-4, negative=5, workers=7)

            tagged_docs_train = self.tagged_docs_pos_train + self.tagged_docs_neg_train
            tagged_docs_test = self.tagged_docs_pos_test + self.tagged_docs_neg_test
            tagged_docs = tagged_docs_train + tagged_docs_test

            self.doc2vec_model.build_vocab(tagged_docs)

            shuffled = list(tagged_docs)
            random.shuffle(shuffled)
            self.doc2vec_model.train(shuffled, total_examples=self.doc2vec_model.corpus_count,
                                     epochs=self.iter)

            self.doc2vec_model.save(model_location + self.doc2vec_model_name)

        else:
            self.doc2vec_model = models.Doc2Vec.load(model_location + self.doc2vec_model_name)

        train_arrays, train_labels = self.data_transformation.create_classifier_arrays(self.doc2vec_model, True,
                                                                                       len(self.tagged_docs_pos_train),
                                                                                       len(self.tagged_docs_neg_train))

        test_arrays, test_labels = self.data_transformation.create_classifier_arrays(self.doc2vec_model, False,
                                                                                     len(self.tagged_docs_pos_test),
                                                                                     len(self.tagged_docs_neg_test))

        clf = LogisticRegression(penalty='l2')
        clf = SVC()
        clf.fit(train_arrays, train_labels)

        # C_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # gamma_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5]
        # param_grid = dict(gamma=gamma_range, C=C_range)
        # cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        # clf = RandomizedSearchCV(SVC(), param_distributions=param_grid, cv=cv, n_iter=5)
        # clf.fit(train_arrays, train_labels)
        #
        # print(clf.best_params_)
        # print(clf.best_estimator_)
        # print(clf.best_score_)

        logging.info("Finished training classifier.")

        # Approach when not training doc2vec on test reviews
        # tvecs = []
        #
        # for i in range(len(self.test_x)):
        #     tdt = TaggedDocument(self.remove_stopwords(self.test_x[i]), ["test_" + str(i)])
        #     tvecs.append(self.doc2vec_model.infer_vector(tdt.words, steps=200))
        #
        # logging.info("Created TaggedDocuments for Training data.")
        # print(classifier.score(test_arrays, test_labels))

        if self.is_real_test:
            file_ids = self.test_x.keys()
            pred = clf.predict(test_arrays)
            self.data_transformation.write_to_file(dict(zip(file_ids, pred)), "doc2vec")
        else:
            v = Visualization(test_labels, clf.predict(test_arrays),
                              "doc2vec - Logistik Regression")
            v.generate()
