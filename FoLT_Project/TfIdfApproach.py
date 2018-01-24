import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from FoLT_Project.Visualization import Visualization


class TfIdfApproach:

    def __init__(self, train_x, train_y, dev_x, dev_y, use_bigramms):

        cv = CountVectorizer(ngram_range=(1, 3)) if use_bigramms else CountVectorizer()
        tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
        # use Stochastic Gradient Descent to determine sentiment
        # which uses without any parameter changes a SVM
        # see http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        # clf = SGDClassifier(loss='hinge', penalty='l2',
        #                     alpha=1e-3, random_state=42,
        #                     max_iter=5, tol=None)
        # use simple Naive Bayes
        clf = MultinomialNB()

        self.pipeline = Pipeline([('vect', cv),
                                  ('tfidf', tfidf),
                                  ('clf', clf)
                                  ])

        self.train_x = train_x
        self.train_y = train_y

        self.dev_x = dev_x
        self.dev_y = dev_y

    def run(self):
        self.pipeline.fit(self.train_x, self.train_y)

        v = Visualization(self.dev_y, self.pipeline.predict(self.dev_x), "TfIdf - no bigrams - MultinomialNB")
        v.generate()
