from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from FoLT_Project.BaseApproach import BaseApproach
from FoLT_Project.Visualization import Visualization


class TfIdfApproach(BaseApproach):

    def __init__(self, train_x, train_y, test_x, test_y, use_bigramms, is_real_test, data_transformation):
        super().__init__()
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

        self.test_x = test_x
        self.test_y = test_y

        self.is_real_test = is_real_test
        self.data_transformation = data_transformation

    def run(self):
        self.pipeline.fit(self.train_x, self.train_y)

        if self.is_real_test:
            file_ids = self.test_x.keys()
            pred = self.pipeline.predict(self.test_x.values())
            self.data_transformation.write_to_file(dict(zip(file_ids, pred)), "TfIdf-b-MNB")
        else:
            v = Visualization(self.test_y, self.pipeline.predict(self.test_x), "TfIdf - no bigrams - MultinomialNB")
            v.generate()
