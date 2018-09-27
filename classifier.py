import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

pokemon_csv = pd.read_csv('Pokemon.csv')

df = pd.DataFrame(pokemon_csv, columns = ['Name', 'Type 1', 'Type 2', 'Total', 'HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary'])


#print(df.apply(lambda x: x.count()))

df = df.drop(['Name'],axis=1)
df = df.dropna(subset=['Type 2'])


i = 0
j = 0
uniqueItem = dict()
uniqueItem2 = dict()

for item in df['Type 1']:
    if item not in uniqueItem:
        uniqueItem[str(item)] = i
        i+=1

for type in df['Type 1']:
    if type in uniqueItem:
        df = df.replace({type:uniqueItem.get(type)})


for item in df['Type 2']:
    if item not in uniqueItem2:
        uniqueItem2[str(item)] = j
        j+=1

for type in df['Type 2']:
    if type in uniqueItem2:
        df = df.replace({type:uniqueItem2.get(type)})


X = np.array(df.iloc[:,0:-1])
Y = np.array([[df['Legendary']]])

Y = Y.reshape(414)

#Naive bayes starts here
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 10)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy score of Naive Bayes: ",accuracy_score(y_test,y_pred))

#SGD Starts here

clf = linear_model.SGDClassifier()
clf.fit(X, Y)

y_pred = clf.predict(X_test)
print("Accuracy score of SDG: ",accuracy_score(y_test,y_pred))

