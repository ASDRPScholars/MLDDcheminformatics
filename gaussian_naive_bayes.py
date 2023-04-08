#imports
import pandas as pd
import math
import os
import numpy as np

final_df = pd.read_csv("./final_combed_set.csv", low_memory=False).drop("Unnamed: 0", axis=1).select_dtypes([np.number])
final_df = final_df.fillna(0)[np.isfinite(final_df).all(1)]

from sklearn.model_selection import train_test_split

#create training and test data
y = final_df['class_label']
del final_df['class_label']
X = final_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def evaluate(tree):
  y_pred = tree.predict(X_test)
  print("Accuracy: ", accuracy_score(y_pred, y_test))
  plot_confusion_matrix(tree, X_test, y_test)  
  plt.show()
  
  print('\n', classification_report(y_test,y_pred))

evaluate (gnb)
