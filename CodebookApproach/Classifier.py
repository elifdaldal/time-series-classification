from Constants import *
import itertools
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
import numpy as np

class Classifier:
    def __init__(self,X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def svc(self):
        clf = GridSearchCV(SVC(), c_tuned_parameters_svc, cv=c_cv_svc, iid=False)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        
        return y_pred
    
    def rfc(self):
        clf = GridSearchCV(RandomForestClassifier(), c_tuned_parameters_rfc, cv=c_cv_rfc, iid=False)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        
        return y_pred
    
    def plot_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
    
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]))

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
        fig.tight_layout()
        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(range(c_activity_count), [l for l in c_labels.values()], rotation = 90)
        plt.yticks(range(c_activity_count), [l for l in c_labels.values()])
        plt.show()
    
        print(classification_report(self.y_test, y_pred, target_names=list(c_labels.values())[:c_activity_count]))