import numpy as np
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier

##########
# Datasets
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

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

regression_MSE = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
print("regression mean square error: " + str(regression_MSE)) # should be 2004.567

# TODO np.c_ Translates slice objects to concatenation along the first axis.



