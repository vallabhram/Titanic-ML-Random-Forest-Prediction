# Titanic-ML-Random-Forest-Prediction

This repository contains code for predicting survival on the Titanic using machine learning techniques. The dataset used in this project is sourced from Kaggle's "Titanic: Machine Learning from Disaster" competition.


**Project Structure**
- train.csv: This file contains the training dataset with features and the target variable, Survived.
- test.csv: This file contains the test dataset with features for which predictions need to be made.
- titanic_ml.py: Python script containing the code for data preprocessing, model training, and predictions.
- README.md: This file contains information about the project, its structure, and how to use the code.


**Requirements**
- Python 3.x
- pandas
- scikit-learn


**Output:**

The script will generate predictions for the test dataset and create an output file named predictions.csv containing PassengerId and corresponding survival predictions.


**Model Overview**

- Random Forest Classifier: The code uses a Random Forest Classifier to predict survival based on features such as Pclass, Sex, SibSp, and Parch. Hyperparameters like the number of estimators and maximum depth are tuned for optimal performance.

- K-Nearest Neighbors (KNN) Classifier: Additionally, a KNN classifier is implemented to demonstrate a different approach to classification. However, due to its simplicity and potential overfitting, it may not yield the best results.
Contributing
Contributions to this project are welcome. If you have any suggestions, feature requests, or bug reports, please open an issue or submit a pull request.
