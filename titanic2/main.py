import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv('train.csv')

print(titanic_df.head())
print(titanic_df.describe())

sns.heatmap(titanic_df.corr(), cmap="YlGnBu")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2)
for train_indices, test_indices in split.split(titanic_df, titanic_df[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic_df.loc[train_indices]
    strat_test_set = titanic_df.loc[test_indices]

plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

plt.show()

print(strat_train_set.info())

from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self,x,y=None):
        return self
    def transform(self,X):
        imputer = SimpleImputer(strategy="mean")
        X ['Age'] = imputer.fit_transform(X [['Age']])
        return X

from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self,X):

        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()

        column_names = ["C", "S", "Q", "N"]


        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        matrix = encoder.fit_transform(X[['Sex']]).toarray()

        column_names = ["Female", "Male"]

        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

            return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")

from sklearn.pipeline import Pipeline

pipeline = Pipeline([("ageimputer", AgeImputer()),
                     ("featureencoder", FeatureEncoder()),
                     ("featuredropper", FeatureDropper())])


strat_train_set = pipeline.fit_transform(strat_train_set)

print(strat_train_set)
print(strat_train_set.info())

from sklearn.preprocessing import StandardScaler

X = strat_train_set.drop(['Survived'], axis=1)
y = strat_train_set['Survived']

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

param_gird = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(clf, param_gird, cv=3, scoring="accuracy", return_train_score=True)
print(grid_search.fit(X_data, y_data))

last_clf = grid_search.best_estimator_
print(last_clf)

strat_test_set = pipeline.fit_transform(strat_test_set)

X_test = strat_test_set.drop(['Survived'],axis=1)
y_test = strat_test_set['Survived']

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
y_data_test = y_test.to_numpy()

print(last_clf.score(X_data_test, y_data_test))

last_data = pipeline.fit_transform(titanic_df)
print(last_data)

X_last = last_data.drop(['Survived'],axis=1)
y_last = last_data['Survived']

scaler = StandardScaler()
X_data_last = scaler.fit_transform(X_last)
y_data_last = y_last.to_numpy()

prod_clf = RandomForestClassifier()

param_gird = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(prod_clf, param_gird, cv=3, scoring="accuracy", return_train_score=True)
print(grid_search.fit(X_data_last, y_data_last))

prod_last_clf = grid_search.best_estimator_
titanic_test_data = pd.read_csv('test.csv')
last_test_data = pipeline.fit_transform(titanic_test_data)
print(last_test_data)

X_last_test = last_test_data
X_last_test = X_last_test.fillna(method="ffill")

scaler = StandardScaler()
X_data_last_test = scaler.fit_transform(X_last_test)

predictions = prod_last_clf.predict(X_data_last_test)
last_df = pd.DataFrame(titanic_test_data['PassengerId'])
last_df['Survived'] = predictions
last_df.to_csv("predictions.csv", index=False)

print(last_df)