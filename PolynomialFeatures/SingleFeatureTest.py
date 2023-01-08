
import numpy as np
import matplotlib.pyplot as plt

X = 10 * np.random.rand(100, 1) - 5
X = np.sort(X, axis=0)
#print(X)
y = 0.5 * X**2 + 2 * X + 2 + np.random.randn(100, 1)
plt.scatter(X, y, c='green', alpha=0.6)
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
X_new = pf.fit_transform(X)
print(X_new)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_new, y)
print(reg.intercept_)
print(reg.coef_)
plt.plot(X, reg.predict(X_new), color='r')
plt.show()