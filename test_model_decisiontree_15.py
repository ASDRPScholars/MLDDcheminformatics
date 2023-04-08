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

#depths_to_test = 24

max_accuracy_depth = 15  #manually inputted, check from graph

accuracy_list = []

final_df = pd.read_csv("../final_combed_set.csv", low_memory=False).drop("Unnamed: 0", axis=1).select_dtypes([np.number])
final_df = final_df.fillna(0)[np.isfinite(final_df).all(1)]

print("Finished filling NA")

output_data = final_df['class_label']
input_data = final_df
del input_data['class_label']

"""
for depth_num in range(1, depths_to_test + 1):
	with open(f"./decision_tree_{depth_num}.pkl", "rb") as f:
		rf = pickle.load(f)
	X_train, X_test, y_train, y_test = model_selection.train_test_split(input_data, output_data, test_size=0.2, random_state=3)

	predictions = rf.predict(X_test)
	accuracy_value = accuracy_score(y_test, predictions)*100
	print('The accuracy value for a depth of ' + str(depth_num) + ' is ' + str(accuracy_value))
	accuracy_list.append(accuracy_value)
"""

with open(f"./decision_tree_{max_accuracy_depth}.pkl", "rb") as f:
	rf = pickle.load(f)

X_train, X_test, y_train, y_test = model_selection.train_test_split(input_data, output_data, test_size=0.2, random_state=3)

predictions = rf.predict(X_test)
accuracy_value = accuracy_score(y_test, predictions)*100
print('The accuracy value for a depth of ' + str(max_accuracy_depth) + ' is ' + str(accuracy_value))

print(rf.feature_importances_)

feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(10).plot(kind='barh')
print("test breaker")
print(feat_importances.nlargest(10))
plt.title("Feature Importances (Top 10) For Decision Tree")
plt.xlabel("Mean Positive SHAP Value (Average Impact on Model)")
plt.ylabel("Feature")
plt.show()
plt.savefig('decision_tree_factor_importance.png')

""" Graph Generation, Not Needed For Gini Calculation
plt.title("Depth of Decision Tree vs Accurary")
plt.xlabel("Depth of Decision Tree")
plt.ylabel("Accurary Percent")
plt.plot(np.arange(1, depths_to_test + 1), accuracy_list, color ="blue")
plt.show()
plt.savefig('accuracy_decision_tree.png')
"""
"""
def gini(array):
	#Calculate the Gini coefficient of a numpy array.
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

    print("Gini: " + str(gini(np.array(accuracy_list))))
"""
