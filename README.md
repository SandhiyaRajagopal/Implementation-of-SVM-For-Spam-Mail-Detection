# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANDHIYA R
RegisterNumber:  212223240146
*/
```
```

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/d4d9e774-6657-4d59-97c0-6f49c25dcab5)
![image](https://github.com/user-attachments/assets/5d3917c4-4f1e-4246-ba20-9aab720bd25f)
![image](https://github.com/user-attachments/assets/2dc990f7-ff05-4a0f-a458-38ed2947a720)
![image](https://github.com/user-attachments/assets/70876684-9870-4814-bbb1-e8d66f8decfc)
![image](https://github.com/user-attachments/assets/18b26772-021b-414a-aa64-3fd069e1c39c)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
