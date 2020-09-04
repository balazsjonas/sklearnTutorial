import numpy as np
from sklearn import linear_model
import pylab as plt

np.random.seed(0)

N = 20
m = 1.2
b = -2.7
x = np.random.uniform(10, 20, N)
x = x.reshape((N,1))
e = np.random.standard_normal(N).reshape((N,1))
y = m * x + b + e

regr = linear_model.LinearRegression()
ridge = linear_model.Ridge(alpha=10)
regr.fit(x,y)
ridge.fit(x,y)
est = regr.predict(x)
est_ridge = ridge.predict(x)

plt.scatter(x, y)
plt.plot(x,est, 'r')
plt.plot(x,est_ridge, 'g')

print(np.std(y - est))

def error(n_samples, noise):
    m = 1.2
    b = -2.7
    x = np.random.uniform(10, 20, n_samples)
    x = x.reshape((n_samples, 1))
    e = np.random.standard_normal(n_samples).reshape((n_samples, 1))
    y = m * x + b + noise * e
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    est = regr.predict(x)
    return np.std(y - est)

N = 100
sample_sizes = np.array([(lambda x:int(x))(tmp) for tmp in np.logspace(1,4, N)])
E = np.zeros(N)
for i in range(len(sample_sizes)):
    E[i] = error(sample_sizes[i], 1)
plt.figure()
plt.plot(sample_sizes, E)
plt.xlabel('sample size')
plt.ylabel('MSE')

noises = np.linspace(0, 2, N)
E2 = np.zeros(N)
for i in range(N):
    E2[i] = error(1000, noises[i])

plt.figure()
plt.plot(noises, E2)
plt.xlabel('"input" noise')
plt.ylabel('"output" noise')
