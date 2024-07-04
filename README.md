# Heart-Disease-Diagnosis-using-ML

                                                               User Manual (Instructions to execute the code)

1.	Data Loading and Exploration:
•	Load the heart disease dataset using ‘pd.read_csv’.
•	Print the first 5 rows of the dataset using ‘head()’ and the last 5 rows using ‘tail()’.
•	Get information about the dataset using ‘info()’ and check for missing values using ‘isnull().sum()’.
•	Describe the statistical measures of the dataset using ‘describe()’.
•	Check the distribution of the target variable using ‘value_counts()’.

2.	Data Preparation:
•	Separate the features (X) and the target variable (Y).
•	Split the dataset into training and testing sets using ‘train_test_split()’.

3.	Model Training and Evaluation (Logistic Regression):
•	Create a Logistic Regression model using ‘LogisticRegression()’.
•	Train the model on the training data using fit().
•	Make predictions on the training and test data using predict().
•	Calculate the accuracy, precision, recall, F1-score, and confusion matrix using appropriate functions from sklearn.metrics.

4.	Model Evaluation (KNN):
•	Create a K-Nearest Neighbors (KNN) model using ‘KNeighborsClassifier()’.
•	Train the KNN model on the training data.
•	Make predictions on the training and test data.
•	Calculate the evaluation metrics as done for Logistic Regression.

5.	Model Evaluation (Naive Bayes):
•	Create a Gaussian Naive Bayes (GNB) model using ‘GaussianNB()’.
•	Train the GNB model on the training data.
•	Make predictions on the training and test data.
•	Calculate the evaluation metrics as done for Logistic Regression.

6.	Model Evaluation (Voting Classifier):
•	Create models for Logistic Regression, Naive Bayes, and Random Forest.
•	Tune hyperparameters for Logistic Regression and Random Forest using ‘GridSearchCV’.
•	Create a Voting Classifier using the best models.
•	Train the Voting Classifier on the training data.
•	Make predictions on the test data and calculate evaluation metrics.

7.	Visualization:
•	Plot the confusion matrices for each model using ‘plot_confusion_matrix’.
•	Plot the ROC curves for each model to visualize the performance using ‘roc_curve’.

8.	Prediction for a New Data Point:
Predict whether a new data point (input_data) indicates heart disease using each model.
