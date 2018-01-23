from FoLT_Project.NaiveBayesApproach import NaiveBayesApproach
from FoLT_Project.Preprocess import Preprocess
from FoLT_Project.TfIdfApproach import TfIdfApproach
from FoLT_Project.corpus_reviews import reviews
from FoLT_Project.Paragraph2Vec import Paragraph2Vec

def main():
    pp = Preprocess(reviews)
    train, dev = pp.split_data()

    # train_words = pp.get_words_without_split(train)
    # dev_words = pp.get_words_without_split(train)
    #
    # nba = NaiveBayesApproach(train_words, dev_words)
    # nba.run()

    # train_x, train_y = pp.get_raw_with_split(train)
    # dev_x, dev_y = pp.get_raw_with_split(dev)
    #
    # tfidf = TfIdfApproach(train_x, train_y, dev_x, dev_y, use_bigramms=True)
    # tfidf.run()

    train_x, train_y = pp.get_words_with_split(train)
    dev_x, dev_y = pp.get_words_with_split(dev)

    p2v = Paragraph2Vec(train_x, train_y, dev_x, dev_y)
    p2v.run()


if __name__ == "__main__":
    main()
