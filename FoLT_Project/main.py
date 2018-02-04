from FoLT_Project.DataTransformation import DataTransformation
from FoLT_Project.NaiveBayesApproach import NaiveBayesApproach
from FoLT_Project.Paragraph2Vec import Paragraph2Vec
from FoLT_Project.TfIdfApproach import TfIdfApproach
from FoLT_Project.corpus_reviews import reviews_train, reviews_test

is_real_test = True


def main():
    dt = DataTransformation(reviews_train, reviews_test, is_real_test=is_real_test)
    train, test = dt.split_data()

    train_words = dt.get_words_without_split(train)
    if is_real_test:
        test_words = dt.get_test_data(test, is_get_raw=False)
    else:
        test_words = dt.get_words_without_split(test)

    nba = NaiveBayesApproach(train_words, test_words, is_real_test=is_real_test, data_transformation=dt)
    nba.run()

    train_x, train_y = dt.get_raw_with_split(train)
    if is_real_test:
        test_x = dt.get_test_data(test, is_get_raw=True)
        test_y = []
    else:
        test_x, test_y = dt.get_raw_with_split(test)

    tfidf = TfIdfApproach(train_x, train_y, test_x, test_y, use_bigramms=True, is_real_test=is_real_test,
                          data_transformation=dt)
    tfidf.run()

    train_x, train_y = dt.get_words_with_split(train)
    if is_real_test:
        test_x = dt.get_test_data(test, is_get_raw=False)
        test_y = []
    else:
        test_x, test_y = dt.get_words_with_split(test)

    p2v = Paragraph2Vec(train_x, train_y, test_x, test_y, is_real_test=is_real_test, data_transformation=dt,
                        build_model=False)
    p2v.run()


if __name__ == "__main__":
    main()
