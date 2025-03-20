import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df =pd.read_csv('/content/sonar.all-data.csv')
df

df.isnull().sum().sum()

df.shape

x = df.drop(columns='R' ,axis=1)
y = df['R']


y.replace({
    'R':0,
    'M':1
}
,inplace = True
)

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_test.shape, x_train.shape,y_train.shape,y_test.shape

lr = LogisticRegression()
lr.fit(x_train,y_train)

x_pred = lr.predict(x_test)
x_accu_score = accuracy_score(x_pred,y_test)

print("Training Accuracy : ",x_accu_score)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)

x_pred = rf.predict(x_test)
x_accu_score = accuracy_score(x_pred,y_test)
print("accu",x_accu_score)

input = (0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
input_as_numpy_array = np.asarray(input)
input_reshape = input_as_numpy_array.reshape(1,-1)
prediction = rf.predict(input_reshape)
print(prediction)

#Pickling

import pickle
filename = 'rocksvsmine.sav' 
pickle.dump(rf,open(filename,'wb'))

loaded_file = pickle.load(open('rocksvsmine.sav','rb'))


input = (0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
input_as_numpy_array = np.asarray(input)
input_reshape = input_as_numpy_array.reshape(1,-1)
prediction = loaded_file.predict(input_reshape)

if prediction[0] == 0:
  print("It is a Rock")
else:
  print("It is a Mine")
