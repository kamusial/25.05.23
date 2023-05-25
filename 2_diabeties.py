import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('diabetes.csv')
print(df.shape)  #wielkość tabeli
print(df.isna().sum())   #puste wartości
print(df.describe().T.to_string())

#wszędzie, gdzie są zera i NA - przepisz średnią (bez zer)
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
    df[col].replace(0, np.NaN, inplace=True)   #zamień zera na NaN
    mean_ = df[col].mean()
    df[col].replace(np.NaN, mean_, inplace=True)
print(df.isna().sum())
print(df.describe().T.to_string())
print(df.iloc[2:4, 4:6])

X = df.iloc[: , :-1]    #wszystkie kolumny, bez ostatniej
y = df.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('dane')
print(df.outcome.value_counts())

df1 = df.query('outcome==0').sample(n=500, random_state=0)
df2 = df.query('outcome==1').sample(n=500, random_state=0)
df3 = pd.concat([df1, df2])

X = df3.iloc[: , :-1]    #wszystkie kolumny, bez ostatniej
y = df3.outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))




