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

The following files contain all of the data optimization code that is necessary to create the final combed dataset for natural and synthetic products that we use as our sample throughout the study. All of these files were placed in a subfolder called all_py/:

1. remove_duplicates.py: this file works with our initial set of SMILES strings, stored in .smi files. It defines a function named remove_sampled_smiles_duplicates that checks through these strings to determine if there are any duplicates between the two datasets, and if they exist, it removes them.
2. rm_dupes_main.py: this file runs the function that was defined in remove_duplicates.py.
3. calc_padel.py: this file imports the PadelPy library to create the dataset for the machine learning models. As input, it receives the SMILES strings of collected natural products and synthetic molecules from the COCONUT and ZINC database. The PadelPy library provides a method to calculate 1876 Padel descriptors for each molecule. The descriptors for each molecule are stored in a separate CSV file, which are split into 400 for natural products and 400 for synthetic.
4. drop_unnamed_columns.py: this file moves through intermediate CSV files of the dataset (after sampling from the 400 nat_csvs/ and 400 syn_csvs/ with sample.py) and drops any unnamed columns that may be present within them. This is because the dataset labeled any columns that did not contain meaningful or interpretable data as “unnamed”. It outputs the final combed dataset that we then split.
5. sample.py: this code randomly samples all the files in nat_csvs/ or syn_csvs/ (depending on the folder it is placed in), creating a dataset.
6. check_null.py: this file checks through the final combed CSV file for any null values. If these are found, it prints out their location. This code was not meant to replace any null values, rather provide us with a metric for how reliable our dataset was and to help troubleshoot any errors that may result from null values.
7. padel_nat_csv_combine_stats.py: this file is meant to drop any columns in the Padel descriptors that are very correlated with one another and therefore largely irrelevant to our analysis. This is done by defining a correlation value “r” and finding columns that have “r = 1”, or maximum correlation. No columns of significance were found, and therefore the values were left unchanged.

The next set of files contains our final combed dataset and all of the code that allows us to do our machine learning analysis. All of these files were placed in a subfolder called padel_ml/:

1. final_combed_set.csv: the final combed CSV that contains all of the data that we will run machine learning analysis on. This is the result of the code that was run in the all_py/ folder.
2. gaussian_naive_bayes.py: The Gaussian Naive Bayes2 machine learning model is utilized to classify the natural products and synthetic molecules in the created dataset. The model is imported from scikit-learn and uses Bayes’ theorem and the assumption of independence among predictors to estimate the probability of each class to guide classification of other algorithms3.
3. mlp_and_logistic_regression.py: The MLP logistic regression model4 is used to classify the natural products and synthetic molecules in the created dataset. The model is imported from sci-kit learn which analyzes the relationships between input features (the PaDEL descriptors) and predicted class based on probabilities with the Sigmoid function.
4. preliminary_decision_tree_full_dataset.py: A Decision Tree model is utilized on a smaller subset of the entire dataset for testing hyperparameters and saving each iteration into a unique pickle file. The model is tested for max_depths in a range from 1 to 25. The decision tree tested root nodes on input data patterns and created leaf nodes as categories to reach a final decision5.
5. preliminary_random_forest_full_dataset.py: A Random Forest model is used on a smaller subset of the entire dataset for testing hyperparameters and saving the iterations in a unique pickle file. The model is tested for max_depths in a range from 1 to 45. The Random Forest is an ensemble learning model that utilized many Decision Trees to process multidimensional data6, 7.
6. test_model.py: This file consists of the test code to ensure that pickle models are loaded properly. A saved decision tree with max_depth 9 is loaded and the accuracy of the model is determined and analyzed on separate test data.
7. test_model_all_decisiontree_15.py: The accuracy and feature importances of a decision tree model with max_depth of 15 is calculated. The max_depth was set to 15 because this hyperparameter resulted in the highest accuracy when tested on a smaller subset of the data in the file: preliminary_decision_tree_full_dataset.py.
8. test_model_all_randomforest_36.py: The accuracy and feature importances of a random forest model with max_depth of 36 is calculated. The max_depth was set to 36 because this hyperparameter resulted in the highest accuracy when tested on a smaller subset of the data in the file:preliminary_random_forest_full_dataset.py.

Not included here but just as important for context are the following folders:
1. nat_csvs/: This subfolder contains all 400 natural product (NP) CSV files, containing the entirety of the NP dataset as derived from Zinc and Coconut. This subfolder also contains the file “nat_csv_avgs.py”, which finds the mean of the values of each of the 1876 descriptors within each file, and compiles these values into a separate CSV file known as “nat_csv_padel_descriptor_ average.csv”, which is also present in this subfolder.
2. syn_csvs/: This subfolder is similar to the nat_csvs/ folder. It contains all 400 synthetic molecule (SM) CSV files, containing the entirety of the synthetic molecule dataset as derived from Zinc and Coconut. This subfolder also contains the file “nat_csv_avgs.py”, as the python file conducts the same function in both folders, with the only difference being that the compiled values are instead placed in the CSV file named “syn_csv_padel_descriptor_average.csv”, which is once again also present within the subfolder.

The usage of the code and files above are outlined as follows:
1. Using our initial .smi files, pulled directly from the Zinc and Coconut databases, rm_dupes_main.py utilizes the function described in remove_duplicates.py ro remove any duplicates within the two datasets.
2. Using the output of step 1, calc_padel.py imports the PadelPy library to output 400 unique CSV files for both natural products and synthetic molecules, all with 1876 Padel descriptors.
    a. These unique CSV files are placed in the folders nat_csvs/ and syn_csvs/ respectively.
3. Using the output of step 2, drop_unnamed_columns.py moves through each of the CSV files and checks for null values, and if found, prints their locations for us to check and fix. 
4. Using the checked CSV files of step 3, sample.py randomly samples them and creates a dataset.
5. Using the sampled dataset outputted in step 4, check_null.py checks through the final combed set for any null values, and padel_nat_combine_stats.py attempts to drop any columns that are directly correlated with each other.
    a. The final output that is the result of this is the final_combed_set.csv, which is stored in the padel_ml/ folder.

Within the padel_ml/ folder:
1. The final_combed_set.csv is run on a variety of models, the first of which is the Gaussian Naive Bayes machine learning model in gaussian_naive_bayes.py.
    a. This outputs Table 2 on our manuscript, which shows the accuracy, precision, and recall of the Gaussian Naive Bayes model.
2. It is then run on the MLP Logistic Regression model with mlp_and_logistic_regression.py.
    a. This outputs Table 1 on our manuscript, which shows the accuracy, precision, and recall of Logistic Regression modeling.
3. Then, the combed set is run through preliminary_decision_tree_ full_dataset.py and preliminary_random_forest_full_dataset.py in order to test hyperparameters and save those iterations in unique pickle files for later use and for our final results.
    a. This outputs Figure 4 on our manuscript, which provides line graphs demonstrating the depth of Random Forest and Decision Tree models and their corresponding accuracies.
4. test_model.py is run to ensure that all the pickle models are loaded properly.
5. Finally, test_model_all_decisiontree_15.py is run in order to obtain the accuracy and feature importances for a decision tree model with a max_depth of 15, which was our highest accuracy depth without overfitting. Similarly, test_model_all_randomforest_36.py was run for random forest at a max depth of 36.
    a. This outputs Figure 5 on our manuscript, which shows the feature importance graphs for these models. Furthermore, it     outputs Table 3, which shows the accuracy, precision, and recall for the Decision Tree model, and Table 4, which shows the accuracy, precision, and recall for Random Forest.
