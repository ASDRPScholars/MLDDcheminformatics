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

'''
Description of section:
1. Starts with creating lists of all natural and all synthetic compounds. Works by taking the last csv in data compilation - the last csv has all the data from
the previous ones anyway.
2. Gives both the dataframes class labels.
3. Concatenates the data sets, and fills all null values with 0.
4. Defines output data and input data.
5. Normalizes all data
'''

'''
all_nat = []
all_nat.append(pd.read_csv("./nat_comb.csv", low_memory=False))
nat_df = pd.concat(all_nat)

all_syn = []
all_syn.append(pd.read_csv("./syn_comb.csv", low_memory=False))
syn_df = pd.concat(all_syn)

print("Finished loading files")

nat_df['class_label'] = 0
syn_df['class_label'] = 1
'''

final_df = pd.read_csv("./final_combed_set.csv", low_memory=False).drop("Unnamed: 0", axis=1).select_dtypes([np.number])
final_df = final_df.fillna(0)[np.isfinite(final_df).all(1)]

print("Finished filling NA")

output_data = final_df['class_label']
input_data = final_df
del input_data['class_label']

#df_norm = final_df[input_data].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

'''
Description of section:
1. Splits the data into x train, x test, y train, and y test.
2. Runs a random forest regressor
3. Creates a feature importance graph, and provides predictions.
'''

X_train, X_test, y_train, y_test = model_selection.train_test_split(input_data, output_data, test_size=0.2, random_state=3)
'''
rf = DecisionTreeClassifier(max_depth=5)
rf.fit(X_train, y_train)
feat_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()
predictions = rf.predict(X_test)
accuracy_value = accuracy_score(y_test, predictions)*100
'''
'''
Description of section:
1. Creates an x, y, and a feature importance list for iteration.
2. Follows same methodology as above, but adds the values to x and y lists
3. Graphs the accuracy
4. Feature importance tries to add lists for each descriptor weight for all max depths into one common list, which will then be iterated trhough.
'''

graph_x = []
graph_y = []
feature_importance_list = []

for y in tqdm(range(1, 25)):
    rf = DecisionTreeClassifier(max_depth=y)
    rf.fit(X_train, y_train)
    with open(f'decision_trees/decision_tree_{y}.pkl', 'wb') as f:
        pickle.dump(rf, f)
    predictions = rf.predict(X_test)
    accuracy_value = accuracy_score(y_test, predictions)*100
    graph_x.append(accuracy_value)
    graph_y.append(y)
    print('The accuracy value for a depth of ' + str(y) + ' is ' + str(accuracy_value))

'''
    features = rf.feature_importances_
    for x in features:
        try:
            feature_importance_list[features.index(x)].append(x)
        except:
            feature_importance_list.append([x])
'''

plt.plot(graph_y, graph_x)
plt.xlabel("Depth of DT")
plt.ylabel("Accuracy")
plt.show()
