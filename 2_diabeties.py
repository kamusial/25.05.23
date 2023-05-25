import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('diabetes.csv')
print(df.shape)  #wielkość tabeli
print(df.isna().sum())   #puste wartości
print(df.describe().T.to_string())




#wszędzie, gdzie są zera i NA - przepisz średnią (bez zer)
for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age']:
