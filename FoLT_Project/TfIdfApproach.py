import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline


class TfIdfApproach:

    def __init__(self, train_x, train_y, dev_x, dev_y, use_bigramms=False):
        cv = CountVectorizer(ngram_range=(1, 3)) if use_bigramms else CountVectorizer()
        # use SVM to determine sentiment
        clf = SGDClassifier(loss='hinge', penalty='l2',
                            alpha=1e-3, random_state=42,
                            max_iter=5, tol=None)
        # use simple Naive Bayes
        # clf = MultinomialNB()

        self.pipeline = Pipeline([('vect', cv),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', clf)
                                  ])

        self.train_x = train_x
        self.train_y = train_y

        self.dev_x = dev_x
        self.dev_y = dev_y

    def run(self):
        self.pipeline.fit(self.train_x, self.train_y)

        predicted = self.pipeline.predict(self.dev_x)
        print("Accuracy for TfIdf:", np.mean(predicted == self.dev_y))
