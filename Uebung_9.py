# Task 9.1
from nltk.corpus import names
import nltk
import random
from nltk.classify import accuracy
import numpy as np


def gender_features(word):
    return {'last_letter': word[-1], 'last_two_letters': word[-2:], 'first_two_letters': word[:2],
            'count': sum([word.count(letter) for letter in list('aeiou')])}


names = [(name, "male") for name in names.words('male.txt')] + [(name, "female") for name in names.words('female.txt')]
random.seed(1)
random.shuffle(names)
train_data = [(gender_features(name), gender) for (name, gender) in names[((len(names) * 9) // 10):]]
test_data = [(gender_features(name), gender) for (name, gender) in names[:((len(names) * 9) // 10)]]

nbc = nltk.NaiveBayesClassifier.train(train_data)
print(accuracy(nbc, test_data))
print("Majority Baseline Improvement:", 0.63 - accuracy(nbc, test_data))
print(nbc.most_informative_features(5))
