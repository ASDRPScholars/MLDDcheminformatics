import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

final_df = pd.read_csv("./final_combed_set.csv", low_memory=False).drop("Unnamed: 0", axis=1).select_dtypes([np.number])
final_df = final_df.fillna(0)[np.isfinite(final_df).all(1)]

from sklearn.linear_model import LogisticRegression

final_combed_set = final_df

final_combed_set.columns
'''
final_combed_set = final_df.drop('smiles',axis=1)
final_combed_set = final_df.dropna()

final_combed_set.isnull().sum()
'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df.drop('class_label', axis = 1), final_combed_set['class_label'])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

score = clf.score(X_test, y_test)
print(score)

from sklearn.metrics import classification_report,confusion_matrix
predictions = clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

import pickle # save model
filename = 'logReg_cheminfo_model.sav'
pickle.dump(clf, open(filename, 'wb'))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state = 1)
mlp.fit(X_train, y_train)

mlp.predict(X_test)

score = mlp.score(X_test, y_test)
print(score)
