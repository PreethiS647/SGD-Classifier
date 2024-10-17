# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Preprocess Data
2.Split the Data
3.Train the Model
4.Evaluate the Model 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Preethi S
RegisterNumber: 212223230157 
*/
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
iris = load_iris()
df=pd.DataFrame(data=iris.data,columns = iris.feature_names)
df['target']=iris.target
print(df.head())
```
```
![image](https://github.com/user-attachments/assets/71e922ba-6b9f-40f0-bbdb-f9a5540b3983)
```
```
X=df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

![image](https://github.com/user-attachments/assets/d4012741-0895-4a10-92f9-685d8c114354)


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

![image](https://github.com/user-attachments/assets/671a61de-57d5-41ed-a97e-ccb51427f0ae)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
