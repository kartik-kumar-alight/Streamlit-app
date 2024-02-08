# Read original dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
iris_df = pd.read_csv("iris.csv")
# iris_df.sample(frac=1, random_state=seed)
# print(iris_df.head(10))
# print(iris_df.columns)
iris_df = iris_df.fillna(0)
# selecting features and target data
X = iris_df[['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']]
y = iris_df[['F_Category']]
# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, stratify=y)
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") # Accuracy: 0.91

joblib.dump(clf, "rf_model.sav")