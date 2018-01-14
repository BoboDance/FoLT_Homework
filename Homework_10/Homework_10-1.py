"""
Original:
Threshold: 500
INFO:root:precision spam: 0.8518518518518519
INFO:root:recall spam: 0.9745762711864406
INFO:root:f-measure spam: 0.9090909090909092

"""

import nltk
import random
import logging
import math
import re
from Homework_10.corpus_mails import mails

logging.basicConfig(level=logging.DEBUG)


def compute_PRF(gold, predicted, class_label):
    """
    :param gold: list of gold labels
    :param predicted: list of predicted labels
    :param class_label: relevant class label
    :return:
    """

    if len(gold) != len(predicted):
        raise ValueError("Shapes do not match!")

    positives = [i for i, j in zip(gold, predicted) if i == j]
    negatives = [i for i, j in zip(gold, predicted) if i != j]

    # true values
    tp = len([i for i in positives if i == class_label])
    tn = len([i for i in positives if i != class_label])

    # false values
    fp = len([i for i in negatives if i != class_label])
    fn = len([i for i in negatives if i == class_label])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return [precision, recall, f_score]


def get_feature_set(fileid, alter_mail_bool, category):
    """
    Extracts features
    :param fileid: MailID
    :param alter_mail_bool: alter this mail or not
    :param category: Spam or NoSpam of given mail
    :return: feature set
    """
    # alter mail before processing and extracting features
    mails_words = alter_mail(mails.words(fileid)) if alter_mail_bool and category == "spam" else mails.words(fileid)

    fd = nltk.FreqDist(token for token in mails_words
                       if token.lower() in top_words)
    features = {}

    # use one percent as a threshold - could also be tuned
    one_percent = math.ceil(fd.N() / 100)

    bigrams = nltk.bigrams(mails_words)

    features['many_exclamation_marks'] = fd['!'] > one_percent
    features['multiple_exclamation_marks'] = len([(w1.lower(), w2.lower())
                                                  for (w1, w2) in bigrams if w1 == "!" and w2 == "!"]) > 0
    features['many_stars'] = fd['*'] > one_percent
    features['many_dollars'] = fd['$'] > one_percent
    features['multiple_dollars'] = len([(w1.lower(), w2.lower())
                                        for (w1, w2) in bigrams if w1 == "$" and w2 == "$"]) > 0
    features['many_minus'] = fd['-'] > one_percent
    features['contains_links'] = len([word for word in fd.keys()
                                      if word.startswith('http')]) > 0
    features['many_repetitions'] = len([(w1.lower(), w2.lower())
                                        for (w1, w2) in bigrams
                                        if w1 == w2 and w1 in top_words and w2 in top_words]) > one_percent
    features['many_out_of_vocab'] = len([w for w in fd.keys()
                                         if w not in brown_words]) > one_percent

    for token in fd.keys():
        features[token] = token

    return features


def evaluate(train_set, test_set, show_features, class_label):
    """
    Evaluates the classifier
    :param train_set:
    :param test_set:
    :param show_features:
    :param class_label:
    :return:
    """
    # Set second param of get_Feature set to True in order to train on modified Spam mails
    # See explanation pdf.
    train_features = [(get_feature_set(fileid, False, category), category) for (fileid, category) in train_set]

    classifier = nltk.NaiveBayesClassifier.train(train_features)
    if show_features:
        classifier.show_most_informative_features(20)

    results = classifier.classify_many([get_feature_set(fileid, True, category) for (fileid, category) in test_set])
    gold = [category for (fileid, category) in test_set]
    return compute_PRF(gold, results, class_label)


def alter_mail(mail):
    """
    Alter email
    :param mail: mail words
    :return: altered mail words
    """
    # randomly sample the middle letters, e.g.
    # studies show this is still readable, but the features do not support this postive
    # (or negative? This depends on the point of view.), so not using this is better.
    # res = []
    #
    #
    # for token in mail:
    #     first = token[0]
    #     last = token[-1]
    #     new = token[1:-1]
    #     res.append(first + ''.join(random.sample(new, len(new)))
    #                + last)
    #     print(res)
    # return res

    mail = list(mail)

    # approach one: add features to identify no spam
    for i in range(200):
        mail.append("linguistics")
        mail.append("languages")
        mail.append("discussion")
        mail.append("language")

    # these words are less frequent therefore add only 100 of them
    for i in range(100):
        mail.append("analysis")
        mail.append("issues")
        mail.append("paper")
        mail.append("development")##

    #  approach two: remove features to identify spam
    for i in range(len(mail)):
        mail[i] = re.sub("http", "", mail[i])  # remove http for from link
        mail[i] = re.sub("!+", "!", mail[i])  # replace more than one exclamation mark
        mail[i] = re.sub("\*+", "", mail[i])  # same as above with stars
        mail[i] = re.sub("\$+", "$", mail[i])  # same as above with dollar
        mail[i] = re.sub("-+", "", mail[i])  # same as above with minus

    return mail


# prepare review data as a list of tuples (vocab, category)
email_data = [(fileid, category)
              for category in mails.categories()
              for fileid in mails.fileids(category)]

random.seed(1)
random.shuffle(email_data)
logging.debug(str(email_data[:10]))

# split in training, develop and test set
train_size = int(0.8 * len(email_data))
train_data, test_data = email_data[:train_size], email_data[train_size:]

train_size = int(0.7 * len(train_data))
train_data, develop_data = train_data[:train_size], train_data[train_size:]

logging.debug("# training develop test: %s %s %s", len(train_data), len(develop_data), len(test_data))

# limit word features to 500 top words, as training with many features takes forever
# try different threshold on how many features work well
for threshold in [500]:
    logging.info("\nThreshold: %d" % threshold)

    fd_train_words = nltk.FreqDist(w.lower() for w in mails.words(fileids=[f for f, c in train_data]))
    top_words = [t for t, _ in fd_train_words.most_common(threshold)]

    brown_words = set([w.lower() for w in nltk.corpus.brown.words()])

    show_most_informative = True
    results = evaluate(train_data, develop_data, show_most_informative, "spam")
    logging.info("precision spam: %s", results[0])
    logging.info("recall spam: %s", results[1])
    logging.info("f-measure spam: %s", results[2])
