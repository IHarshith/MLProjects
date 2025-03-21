import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

chunk_size = 1000
for chunk in pd.read_csv('/content/Fake.csv', chunksize=chunk_size):
  df_fake = chunk
for chunk in pd.read_csv('/content/True.csv', chunksize=chunk_size):
  df_true = chunk


df_fake.head()

df_fake['class'] = 0
df_true['class'] = 1

df_fake.shape,df_true.shape

df= pd.concat([df_fake, df_true], axis=0)
df.head()

df['class'].unique()

df = df.drop(['title','subject','date'], axis=1)

df.isnull().sum()

port_stem = PorterStemmer()

def stemming(title):
  stemmed_content = re.sub('[^a-zA-Z]',' ',title)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

df['text'] = df['text'].apply(stemming)

x = df['text'].values
y = df['class'].values

x.shape,y.shape

vectorizer = TfidfVectorizer()
vectorizer.fit(x)

x = vectorizer.transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

lr = LogisticRegression()
lr.fit(x_train,y_train)

x_Pred = lr.predict(x_train)
x_train_acc = accuracy_score(x_Pred,y_train)
print("Accuracy Score :",x_train_acc)

input = x_test[1]

prediction = lr.predict(input)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

import pickle
filename = 'fakenewsdetect'
pickle.dump(lr,open(filename,'wb'))

loaded_model = pickle.load(open('fakenewsdetect','rb'))

input = x_test[1]

prediction = loaded_model.predict(input)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')
