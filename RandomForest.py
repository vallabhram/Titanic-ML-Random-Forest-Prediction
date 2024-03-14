import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score

test = pd.read_csv("C:/Vallabh/SELF/Misc/Titanic ML/test.csv")
train = pd.read_csv("C:/Vallabh/SELF/Misc/Titanic ML/train.csv")

train.groupby('Sex')['Survived'].value_counts()
len(train[train['Survived'] == 1]) / len(train)
len(train[(train['Survived'] == 1) & (train['Sex'] == 'male')]) / len(train[train['Sex'] == 'male']) 
len(train[(train['Survived'] == 1) & (train['Sex'] == 'female')]) / len(train[train['Sex'] == 'female']) 

y = train['Survived']
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

rfmodel = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)
rfmodel.fit(X,y)

predictions = rfmodel.predict(X_test)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
print(output)

accuracy = accuracy_score(y, rfmodel.predict(X))
print("Accuracy:", accuracy)








