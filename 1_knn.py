import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('iris.csv')
#print(df.to_string())   #wypisz wszystko
#print(df.head(15))
print(df.columns)
print(df.sepallength)  #kolumna
print(df['class'])   #kolumna
print(df['class'].value_counts())



