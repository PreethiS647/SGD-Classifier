# Implementation of SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import essential libraries like sklearn, numpy and pandas. load the dataset (CSV file) and explore it to understand the feature-target structure\
2. Use train_test_split() from sklearn.model_selection to divide the data. Split the dataset into training and testing sets to train the model and evaluate its performance
3. Standardize features using StandardScaler to improve covergence during gradient descent. Fit the scaler on the training data and transform both training and testing data accordingly
4. Use SGDClassifier from sklearn.linear-model and fit the model on the training data. Specify the loss function and train the model using SGD
5. Use the trained model to make predictions on the test data using predict(). Store these predictions to compare with actual test labels and evaluate the model's performance
6. Calculate accuracy. This tells how well the model classifies the test data correctly
7. Use confusion-matrix() to generate confusion matrix. This matrix show the count of True positives, False positives, True negatives and False negatives to analyze the classification performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Preethi S
RegisterNumber:  212223230157

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df=pd.DataFrame(data=iris.data,columns = iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

*/
```

## Output:
# Head

![Screenshot 2024-09-20 132655](https://github.com/user-attachments/assets/42c2c25f-b995-4ea4-a740-146df4243967)

# Accuracy and Confusion matrix

![Screenshot 2024-09-20 132705](https://github.com/user-attachments/assets/7c84dade-7112-4300-9d6e-fe26f737b7c6)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
