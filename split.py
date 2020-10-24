import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

data=pd.read_csv('HAR_N.csv')
data.head()
a=data
a_train, a_test = train_test_split(a, test_size=0.30, random_state=0)

a_train.to_csv('HAR_N_70.csv',index=False)
a_test.to_csv('HAR_N_30.csv',index=False)