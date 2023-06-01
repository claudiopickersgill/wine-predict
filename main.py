from scripts.base import base
from scripts.split import split
from scripts.train import train
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import utils_config
config = utils_config.load_config("./config.json")


config = [
    (SVC, {'kernel': 'rbf'}),
    (SVC, {'kernel': 'rbf', 'gamma': 2}),
    (SVC, {'degree': 3, 'kernel': 'poly'} ),
    (SVC, {'degree': 5, 'kernel': 'poly'} ),
    (SVC, {'degree': 10, 'kernel': 'poly'} ),
    (LogisticRegression, {}),
    (DecisionTreeClassifier, {'min_samples_leaf': 50})]

# na função base dá pra usar 'red' ou 'white'
var = 'white'
X, y = base(var)
X_train_cv, X_test, y_train_cv, y_test = split(X, y)

for model_class, settings in config:
    train(var, X_train_cv, y_train_cv, X_test, y_test, model_class, settings)