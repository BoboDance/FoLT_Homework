import logging
import os
import random

from gensim import corpora, models, similarities
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_location = os.path.dirname(os.path.realpath(__file__)) + "/models"


class Paragraph2Vec:

    def __init__(self, train_x, train_y, dev_x, dev_y):

        self.iter = 20
        self.stop_words = stopwords.words("english")

        self.train_x = train_x
        self.train_y = train_y

        self.dev_x = dev_x
        self.dev_y = dev_y

        self.corpus = None
        self.dictionary = None
        self.tfidf_model = None
        self.lsi_model = None
        self.index = None

        self.doc2vec_model = None
        self.tagged_docs_pos_train = None
        self.tagged_docs_neg_train = None
        self.tagged_docs_pos_test = None
        self.tagged_docs_neg_test = None

    def run(self, build_model=True):

        self.tagged_docs_pos_train, self.tagged_docs_neg_train = self.transform_sents(self.train_x, self.train_y, False)
        self.tagged_docs_pos_test, self.tagged_docs_neg_test = self.transform_sents(self.dev_x, self.dev_y, True)

        if build_model or not os.path.exists(model_location + '/reviews.doc2vec'):
            self.doc2vec_model = models.Doc2Vec(min_count=1, window=10, size=400, sample=1e-4, negative=5, workers=7)

            tagged_docs_train = self.tagged_docs_pos_train + self.tagged_docs_neg_train
            tagged_docs_test = self.tagged_docs_pos_test + self.tagged_docs_neg_test

            self.doc2vec_model.build_vocab(tagged_docs_train + tagged_docs_test)

            for epoch in range(self.iter):
                shuffled = list(tagged_docs_train + tagged_docs_test)
                random.shuffle(shuffled)
                self.doc2vec_model.train(shuffled, total_examples=self.doc2vec_model.corpus_count,
                                         epochs=self.iter)

            self.doc2vec_model.save(model_location + '/reviews.doc2vec')

        else:
            self.doc2vec_model = models.Doc2Vec.load(model_location + '/reviews.doc2vec')

        train_arrays = []
        train_labels = []

        for i in range(len(self.tagged_docs_pos_train)):
            train_arrays.append(self.doc2vec_model.docvecs['train_pos_' + str(i)])
            train_labels.append(1)

        for i in range(len(self.tagged_docs_neg_train)):
            train_arrays.append(self.doc2vec_model.docvecs['train_neg_' + str(i)])
            train_labels.append(0)

        test_arrays = []
        test_labels = []

        for i in range(len(self.tagged_docs_pos_test)):
            test_arrays.append(self.doc2vec_model.docvecs['test_pos_' + str(i)])
            test_labels.append(1)

        for i in range(len(self.tagged_docs_neg_test)):
            test_arrays.append(self.doc2vec_model.docvecs['test_neg_' + str(i)])
            test_labels.append(0)

        classifier = LogisticRegression()
        classifier.fit(train_arrays, train_labels)

        print(classifier.score(test_arrays, test_labels))

    def calc_simalarity(self, build_model):
        self.get_models(build_model)

        texts_dev = self.remove_stopwords_corpus(self.dev_x)

        # for sent in texts_dev:
        vec_bow = self.dictionary.doc2bow(texts_dev[0])
        vec_lsi = self.lsi_model[vec_bow]
        similar = self.index[vec_lsi]
        print(zip(similar, self.dev_y))
        similar = sorted(enumerate(similar), key=lambda item: -item[1])
        print(similar)

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

    def get_models(self, build_model):
        if build_model:
            texts_train = self.remove_stopwords_corpus(self.train_x)
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

    def transform_sents(self, sents, labels, is_test):

        neg_ctr = 0
        pos_ctr = 0
        pos = []
        neg = []
        for i, tokens in enumerate(sents):
            sent_class = labels[i]

            if sent_class == "neg":
                neg.append(TaggedDocument(self.remove_stopwords(tokens),
                                          [("train_" if not is_test else "test_") + sent_class + '_%s' % neg_ctr]))

                neg_ctr += 1
            else:
                pos.append(TaggedDocument(self.remove_stopwords(tokens),
                                          [("train_" if not is_test else "test_") + sent_class + '_%s' % pos_ctr]))
                pos_ctr += 1

        logging.info("Finished formatting documents.")
        return pos, neg
