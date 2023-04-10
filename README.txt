Key Molecular Descriptors Distinguishing Between Synthetic and Natural Products

Nathaniel Thomas [1, 9, 10], Sweekrit Bhatnagar [2, 10], Anya Iyer [3, 10], Avirral Agarwal [4, 10], Niranjana Sankar [5, 10], Manav Bhargava [6, 10], Spencer Ye [1, 9, 10], Edwin Li [7, 10], Ritam Nandi [8, 10], Edward Njoo [9], Robert Downing [10] 

[1] Homestead High School 
[2] Mountain House High School 
[3] Dougherty Valley High School 
[4] Carlmont High School 
[5] Monta Vista High School 
[6] Los Altos High School 
[7] Foothill High School 
[8] Mission San Jose High School 
[9] Department of Chemistry, Biochemistry, & Physics, Aspiring Scholars Directed Research Program, Fremont, CA 
[10] Department of Computer Science & Engineering, Aspiring Scholars Directed Research Program, Fremont, CA

Descriptions of the files listed:

calc_padel.py
This code imports the PadelPy library to create the dataset for the machine learning models. As input, it receives the SMILES strings of collected natural products and synthetic molecules from the COCONUT and ZINC database. The PadelPy library provides a method to calculate 1876 Padel descriptors for each molecule. The descriptors for each molecule are stored in a separate CSV file.

gaussian_naive_bayes.py
The Gaussian Naive Bayes2 machine learning model is utilized to classify the natural products and synthetic molecules in the created dataset. The model is imported from scikit-learn and uses Bayes’ theorem and the assumption of independence among predictors to estimate the probability of each class to guide classification of other algorithms3.

mlp_and_logistic_regression.py
The MLP logistic regression model4 is used to classify the natural products and synthetic molecules in the created dataset. The model is imported from sci-kit learn which analyzes the relationships between input features (the PaDEL descriptors) and predicted class based on probabilities with the Sigmoid function.

preliminary_decision_tree_full_dataset.py
A Decision Tree model is utilized on a smaller subset of the entire dataset for testing hyperparameters and saving each iteration into a unique pickle file. The model is tested for max_depths in a range from 1 to 25. The decision tree tested root nodes on input data patterns and created leaf nodes as categories to reach a final decision5.

preliminary_random_forest_full_dataset.py
A Random Forest model is used on a smaller subset of the entire dataset for testing hyperparameters and saving the iterations in a unique pickle file. The model is tested for max_depths in a range from 1 to 45. The Random Forest is an ensemble learning model that utilized many Decision Trees to process multidimensional data6, 7.

test_model.py
This file consists of the test code to ensure that pickle models are loaded properly. A saved decision tree with max_depth 9 is loaded and the accuracy of the model is determined and analyzed on separate test data..

Test_model_all_decisiontree_15.py
The accuracy and feature importances of a decision tree model with max_depth of 15 is calculated. The max_depth was set to 15 because this hyperparameter resulted in the highest accuracy when tested on a smaller subset of the data in the file: preliminary_decision_tree_full_dataset.py.

test_model_all_randomforest_36.py
The accuracy and feature importances of a random forest model with max_depth of 36 is calculated. The max_depth was set to 36 because this hyperparameter resulted in the highest accuracy when tested on a smaller subset of the data in the file:preliminary_random_forest_full_dataset.py.
