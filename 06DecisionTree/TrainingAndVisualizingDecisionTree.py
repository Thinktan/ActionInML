
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

print(y)

from sklearn.tree import export_graphviz
import os
import platform

OUTPUT_ROOT_DIR = "~/Downloads" # macos

print(platform.system().lower())
print(iris.feature_names[2:])
print(iris.target_names)


CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(OUTPUT_ROOT_DIR, CHAPTER_ID, "images")
print(IMAGES_PATH)

os.makedirs(IMAGES_PATH, exist_ok=True)
print(os.path.join(IMAGES_PATH, "iris.dot"))

export_graphviz(
    tree_clf,
    #out_file=os.path.join(IMAGES_PATH, "iris.dot"),
    out_file=os.path.join("iris.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)