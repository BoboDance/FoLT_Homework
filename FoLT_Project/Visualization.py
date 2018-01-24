import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class Visualization:

    def __init__(self, y_test, y_pred, approach_name, class_names=("pos", "neg")):
        self.y_test = y_test
        self.y_pred = y_pred
        self.class_names = class_names
        self.approach_name = approach_name

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def generate(self, normalize=False):

        prfs = precision_recall_fscore_support(self.y_test, self.y_pred)

        print(self.approach_name)
        for i, cl in enumerate(self.class_names):
            print("Precision for " + cl + ":", prfs[0][i])
            print("Recall for " + cl + ":", prfs[1][i])
            print("F-Measure for " + cl + ":", prfs[2][i])

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)
        np.set_printoptions(precision=2)

        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.class_names, normalize=normalize,
                                   title='Confusion matrix for ' + self.approach_name)

        plt.show()
