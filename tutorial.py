import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pylab as plt

##########
# Datasets
print("datasets")
iris = datasets.load_iris()
digits = datasets.load_digits()
# https://scikit-learn.org/stable/tutorial/statistical_inference/settings.html
# first axis is the samples axis
# second is the features axis
print("iris.data.shape: " + str(iris.data.shape))
print(str(iris.data.shape[0]) + " observations")
print(str(iris.data.shape[1]) + " features")

# reshape
# digits.images.shape (1797,8,8) -> (1797, 64) # 64 features
# digits.data.shape
# digits.images.reshape((digits.images.shape[0], -1))

# classification -> set of finite labels (int or string)
# regression -> predict a continuous target variable

##########
# k-nearest neighbors classifier:
print("k nearest neighbors")
np.random.seed(0)
iris_X, iris_y = datasets.load_iris(return_X_y=True)
np.unique(iris_y)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

iris_y_predicted = knn.predict(iris_X_test)
hit_ratio = np.mean(iris_y_test == iris_y_predicted)
print("hit ratio: " + str(hit_ratio))

## Linear Regression
# https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.ht ml#linear-regression
print("linear regression")
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

regression_MSE = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
print(
  "regression mean square error: " + str(regression_MSE))  # should be 2004.567

# TODO np.c_ Translates slice objects to concatenation along the first axis.

##########
# classification
# https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#classification
print("classification")
log = linear_model.LogisticRegression(C=1e5)
log.fit(iris_X_train, iris_y_train)
# The C parameter controls the amount of regularization in the LogisticRegression object: a large value for C results in less regularization.
print("train score: " + str(log.score(iris_X_train, iris_y_train)))
print("test score: " + str(log.score(iris_X_test, iris_y_test)))

###
# model selection
# https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
print("model selection")
X_digits, y_digits = datasets.load_digits(return_X_y=True)
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100])
k_fold = KFold(n_splits=5)
print("cross validation scores: " +
      str(cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)))
Cs = np.logspace(-5, 0, 10)
scores = np.zeros_like(Cs)
for i in range(10):
  svc = svm.SVC(C=Cs[i], kernel='rbf')
  scores[i] = np.mean(cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1))
plt.semilogx(Cs, scores)
plt.show()

# TODO https://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html
