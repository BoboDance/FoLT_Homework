from Exercises.corpus_mails import mails


# Task 10.1
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

    return {"precision": precision, "recall": recall, "f_score": f_score}


# Task 10.2
gold = ["Spam", "Spam", "NoSpam", "NoSpam", "Spam", "Spam", "NoSpam", "NoSpam", "Spam"]
pred = ["NoSpam", "Spam", "NoSpam", "Spam", "Spam", "NoSpam", "Spam", "NoSpam", "Spam"]
print(compute_PRF(gold, pred, "Spam"))

# Task 10.3

print(mails.categories())
print(len(mails.fileids('spam')))
print(len(mails.fileids('nospam')))

# Task 10.4
# length
# typos
# upper case letter
# token freq
