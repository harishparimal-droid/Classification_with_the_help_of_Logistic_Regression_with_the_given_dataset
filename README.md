# Classification_with_the_help_of_Logistic_Regression_with_the_given_dataset
Loaded and cleaned breast cancer data, encoded the target, split and standardized features, then trained a logistic regression model. Evaluated with confusion matrix, precision, recall, ROC-AUC, and plotted ROC curve. Explained sigmoid function and threshold tuning for classification.

Binary Classification with Logistic Regression
This project implements a binary classifier using logistic regression to predict breast cancer diagnosis (malignant or benign) based on real patient data.

Project Overview
Dataset: Breast cancer diagnostic data (features from digitized images of fine needle aspirate).

Goal: Classify tumors as malignant (1) or benign (0) using logistic regression.

Workflow
Data loaded and cleaned (removed IDs and missing columns).

Encoded labels for binary classification.

Split data into training and test sets (80/20), then standardized features.

Trained logistic regression model using scikit-learn.

Evaluated using confusion matrix, precision, recall, and ROC-AUC score.

Plotted and interpreted ROC curve for model performance.

Explained sigmoid activation and the effect of classification threshold.

Results
Confusion Matrix:
[[70 1]
[ 2 41]]

Precision: 0.976

Recall: 0.953

ROC-AUC: 0.997

Files
data.csv — Input dataset

main.py — Implementation code

Logistic_Regression_Graph.jpg — ROC curve visualization

Key Concepts
Logistic Regression: Used for binary classification, outputting probabilities via the sigmoid function.

Evaluation Metrics: Confusion matrix, precision, recall, ROC-AUC.

Threshold Tuning: Adjusting probability cutoff affects precision/recall balance.

Usage
Install dependencies:
pip install pandas scikit-learn matplotlib

Run the code:
python main.py

Review results and ROC graph.
