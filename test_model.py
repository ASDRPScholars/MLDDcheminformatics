from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from tqdm import tqdm
import numpy as np

depth_to_test = 9

final_df = pd.read_csv("../final_combed_set.csv", low_memory=False).drop("Unnamed: 0", axis=1).select_dtypes([np.number])
final_df = final_df.fillna(0)[np.isfinite(final_df).all(1)]

print("Finished filling NA")

output_data = final_df['class_label']
input_data = final_df
del input_data['class_label']

with open(f"./decision_tree_{depth_to_test}.pkl", "rb") as f:
    rf = pickle.load(f)
X_train, X_test, y_train, y_test = model_selection.train_test_split(input_data, output_data, test_size=0.2, random_state=3)

predictions = rf.predict(X_test)
accuracy_value = accuracy_score(y_test, predictions)*100
print('The accuracy value for a depth of ' + str(depth_to_test) + ' is ' + str(accuracy_value))
