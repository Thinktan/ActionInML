from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from PolynomialFeaturesUtils import plot_predictions, plot_dataset


from sklearn.svm import SVC

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
#print(y)

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])

poly_kernel_svm_clf.fit(X, y)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()