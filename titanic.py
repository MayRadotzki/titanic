import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

titanic_data = pd.read_csv('titanic_data.csv')
titanic = titanic_data.copy()

mean = titanic['Age'].mean()
titanic['Age'].fillna(mean, inplace=True)

most_frequent = titanic['Embarked'].value_counts().idxmax()
titanic['Embarked'].fillna(most_frequent, inplace=True)
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

titanic['Sex'] = titanic['Sex'].map({'female': 1, 'male': 0})

X = titanic.drop(columns=['Survived'])
y = titanic['Survived'].values

#create new knn model
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, int(int(titanic.shape[0]) / 2))}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, iid=True)

#fit model to data
knn_gscv.fit(X, y)

#check top performing n_neighbors value
optimal_k=knn_gscv.best_params_

#check mean score for the top performing value of n_neighbors
acc=knn_gscv.best_score_

print('Best K: {}, Best Accuracy: {}'.format(optimal_k, acc))