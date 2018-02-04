# Prerequisites #

Please have the following installed:

- Other:
    - C-Compiler

- Python Packages:
    - numpy
    - sklearn
    - gensim

Please also note that retraining the doc2vec model can take really long.
I pretrained a model and dumped it (I hope I can upload it, it is quite large).
However, you should change the boolean _build_model=False_ in the main, if you do not want to waint 5h.

My own approach also takes really long and probably should be commented as well.
Building the bigrams of the whole corpus takes quite a bit.
But feel free to execute it. :D

The training data should be available in your nltk_data folder in _aclImdb/train_ as provided in moodle.
The test data should be available in your nltk_data folder in _aclImdb/test_ as provided in moodle.

The current version does not print the result to the console but instead places a zip in the project folder.
Results are only displayed when using _is_real_test=False_.

The following scores where achieved on the training data.
The results of the test can be found in Codalab.

# Evaluation scores: #

## Homework 9/NaiveBayesApproach ##
- Accuracy avg: 0.891

## TfIdf Approach ##

### No Bigrams - MultinomialNB ###
- Precision for pos: 0.866770549279
- Recall for pos: 0.883287018658
- F-Measure for pos: 0.874950845458
- Precision for neg: 0.879161528977
- Recall for neg: 0.86215235792
- F-Measure for neg: 0.870573870574

### No Bigrams - SGDClassifier ###
- Precision for pos: 0.91847826087
- Recall for pos: 0.805081381501
- F-Measure for pos: 0.858049502856
- Precision for neg: 0.824140401146
- Recall for neg: 0.927448609432
- F-Measure for neg: 0.872747961312

### Bigrams - MultinomialNB ###
- Precision for pos: 0.889277389277
- Recall for pos: 0.908693926161
- F-Measure for pos: 0.898880816807
- Precision for neg: 0.905193734542
- Recall for neg: 0.885126964933
- F-Measure for neg: 0.895047890768

### Bigrams - SGDClassifier ###
- Precision for pos: 0.874259102456
- Recall for pos: 0.819769749901
- F-Measure for pos: 0.846138086458
- Precision for neg: 0.827899924185
- Recall for neg: 0.880290205562
- F-Measure for neg: 0.853291658527

## Doc2Vec ##

### Logistic Regression ###
- Precision for pos: 0.885034288019
- Recall for pos: 0.870980547836
- F-Measure for pos: 0.877951180472
- Precision for neg: 0.87108290361
- Recall for neg: 0.885126964933
- F-Measure for neg: 0.878048780488

### SVM ###
- Precision for pos: 0.8821385176184691
- Recall for pos: 0.8646288209606987
- F-Measure for pos: 0.8732959101844427
- Precision for neg: 0.8652706440142236
- Recall for neg: 0.8827085852478839
- F-Measure for neg: 0.87390263367917
- Accuracy avg: 0.8736