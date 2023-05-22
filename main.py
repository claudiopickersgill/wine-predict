from .scripts.base import base
from .scripts.split import split
from .scripts.train import train
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

X, y = base('red')
X_train_cv, X_test, y_train_cv, y_test = split(X, y)
train(X_train_cv, y_train_cv, X_test, y_test, LogisticRegression)
train(X_train_cv, y_train_cv, X_test, y_test, DecisionTreeClassifier)