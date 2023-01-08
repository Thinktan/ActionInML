import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X1 = 10 * np.random.rand(100, 1) - 5
X2 = 2 * np.random.rand(100, 1) - 9

y = 2 * X1 + 3 * X2 + 4 * X1**2 + 5 * X1*X2 + 6 * X2**2 + 7 + np.random.rand(100, 1)

X = pd.DataFrame(np.c_[X1, X2])
X.columns = ['x1', 'x2']


# 生成新特征
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
X_new = pf.fit_transform(X)
# print(X_new)
columns_list = pf.get_feature_names_out()
# print(columns_list)

features = pd.DataFrame(X_new, columns=columns_list)
#print(feature)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(features, y)
print(reg.intercept_)
print(reg.coef_)
